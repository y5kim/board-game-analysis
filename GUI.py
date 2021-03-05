from tkinter import * #gui module
from tkinter import ttk
from PIL import ImageTk,Image


import wget #file and url handling
import os
import webbrowser


#preprocessing stuff
import ast
import os
import itertools
from collections import Counter

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import preprocessing as pre

#recommender stuff
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import pairwise_distances
from word_cloud import get_stop_words

import recommendation_utils as recommender
import filter as ft

#preprocessing stuff
# Read data
games = pd.read_csv("Data/games_detailed_info.csv", index_col=0) # review stats
n_rows, n_cols = games.shape
# 1. Remove columns with > 20% of NA values

key_columns = pre.keep_columns_with_few_na(games)
# 2. Remove redundant/unnecesary columns
unnecessary_columns = ["type", "image", "suggested_num_players", "suggested_playerage",
                       "suggested_language_dependence"]
readd_columns = ["boardgameexpansion"]
key_columns = [x for x in key_columns if x not in unnecessary_columns]
key_columns.extend(readd_columns)
# 3. Rename confusing column names
games = games.loc[:,key_columns]
games.rename(columns={"primary": "name", "usersrated": "numratings", "average": "avgrating",
                      "boardgamecategory": "category", "boardgamemechanic": "mechanic",
                      "boardgamedesigner": "designer", "boardgamepublisher": "publisher",
                      "bayesaverage": "bayesavgrating", "Board Game Rank": "rank",
                      "stddev": "stdrating", "median": "medianrating",
                      "owned": "numowned", "trading": "numtrades", "wanting":"numwants",
                      "wishing": "numwishes", "averageweight":"complexity"}, inplace=True)

names = games['name'].values
ids = games['id'].values
thumbs =games['thumbnail'].values

#recommender stuff

# 4. Parse columns with list values
# Convert list of strings to list

for list_col in ["category", "mechanic", "designer", "publisher"]:
    games[list_col] = games[list_col].apply(lambda x: ast.literal_eval(x) if not(pd.isna(x)) else [])


list_colnames = ["category", "mechanic", "designer", "publisher"]
#games = pre.parse_list_columns(games, list_colnames)
cat_list = list(ft.available_choices(games,"category"))
mech_list = list(ft.available_choices(games, "mechanic"))
des_list = list(ft.available_choices(games, "designer"))
pub_list = list(ft.available_choices(games, "publisher"))


# 5. Keep top 10000 games and create encoded columns

top_10k_games = games[games['rank'] <= 10000].reset_index(drop=True)
# Encode multi-categorical columns
recommendation_df = top_10k_games.copy()
mechanic_encoding = recommender.add_encoded_column(recommendation_df, 'mechanic')
category_encoding = recommender.add_encoded_column(recommendation_df, 'category')
publisher_encoding = recommender.add_encoded_column(recommendation_df, 'publisher', 50, ['(Public Domain)', '(Unknown)', '(Web published)'])
designer_encoding = recommender.add_encoded_column(recommendation_df, 'designer', 20, ['(Uncredited)'])
# Generate counter columns
recommender.add_item_counts_column(recommendation_df, 'boardgameexpansion')

# 6. Generate similarity matrices using features of interest

similarity_cols = ['description'
                   , 'category_encoded'
                   , 'mechanic_encoded'
                   , 'publisher_encoded'
                   , 'designer_encoded'
                   , 'complexity'
                  ]

similarity_types = ['text'
                    , 'one_hot'
                    , 'one_hot'
                    , 'one_hot'
                    , 'one_hot'
                    , 'scalar'
                   ]

similarity_weights = [0.1, 0.25, 0.25, 0.1, 0.1, 0.2]

similarity_matrices = recommender.get_similarity_matrices(recommendation_df, similarity_cols, similarity_types)
# similarity_matrices = [[]]

###################################################

default_img = 'default_image.jpg'

class Game:
    """an object to easily link a game name to it's id and thumbnail"""
    myName = ''
    ID = 0
    thumbnail = ''

    def __init__(self, name):
        if name == ' ':
            self.myName = ''
            self.ID = 0
            self.thumbnail = ''
            self.my_img = ImageTk.PhotoImage(Image.open(default_img))
        else:
            self.myName = name
            self.ID = ids[np.where(names == self.myName)][0]
            self.thumbnail = thumbs[np.where(names == self.myName)][0]
            my_img_file = wget.download(self.thumbnail)
            print(' ')
            image_files.append(my_img_file)
            self.my_img = ImageTk.PhotoImage(Image.open(my_img_file))

    def set(self,name):
        if name == ' ':
            self.myName = ''
            self.ID = 0
            self.thumbnail = ''
            self.my_img = ImageTk.PhotoImage(Image.open(default_img))
        else:
            self.myName = name
            self.ID = ids[np.where(names == self.myName)][0]
            self.thumbnail = thumbs[np.where(names == self.myName)][0]
            my_img_file = wget.download(self.thumbnail)
            print(' ')
            image_files.append(my_img_file)
            self.my_img = ImageTk.PhotoImage(Image.open(my_img_file))

###################################################
def openLink(gameID):
    url ='https://boardgamegeek.com/boardgame/'+str(gameID)
    webbrowser.open(url)  # Go to example.com
    return

def search_switcher(search_type):
    global mode
    if search_type == 'similar':
        filter_frame.grid_forget()
        user_frame.grid_forget()
        similar_frame.grid(row =0, column =1, columnspan = 2, rowspan = 2)
        exclude.grid(row = 1, column = 0)
        game_table.grid(row=3, column=1, columnspan=2, pady=10)
        exclude_table.grid(row=3, column=0, columnspan=1, pady=10)
        mode = 0
    elif search_type == 'filter':
        exclude.grid_forget()
        exclude_table.grid_forget()
        game_table.grid_forget()
        similar_frame.grid_forget()
        user_frame.grid_forget()
        filter_frame.grid(row =0, column =1, columnspan = 2, rowspan = 2)
        mode = 1
    else :
        exclude.grid_forget()
        exclude_table.grid_forget()
        game_table.grid_forget()
        similar_frame.grid_forget()
        filter_frame.grid_forget()
        user_frame.grid(row =0, column =1, columnspan = 2, rowspan = 2)
        mode =2
    return

def search_handeler():
    print(mode)
    global games
    if mode == 0:
        name = auto.get()
        if var1.get() == 0:
            arg_list.append(name)
            weight = w.get()
            game_weights.append(float(weight))
            game_table.insert('', 'end', values=(name, weight))
        else:
            excluded.append(name)
            exclude_table.insert('', 'end', values=(name,))

        #call recommender and put in recs list
        print(arg_list)
        print(game_weights)
        print(excluded)
        recs = recommender.recommend_games(recommendation_df, arg_list, similarity_matrices,game_weights=game_weights, similarity_weights=similarity_weights, num_games=15, exclude=excluded)
        for i in range(9):
            game_objs[i].set(recs[i])

            if i==0:
                button0['image'] = game_objs[i].my_img
            elif i==1:
                button1['image'] = game_objs[i].my_img
            elif i==2:
                button2['image'] = game_objs[i].my_img
            elif i==3:
                button3['image'] = game_objs[i].my_img
            elif i==4:
                button4['image'] = game_objs[i].my_img
            elif i==5:
                button5['image'] = game_objs[i].my_img
            elif i==6:
                button6['image'] = game_objs[i].my_img
            elif i==7:
                button7['image'] = game_objs[i].my_img
            elif i==8:
                button8['image'] = game_objs[i].my_img


    elif (mode == 1):
        print(cat.get())
        print(mech.get())
        print(des.get())
        print(pub.get())
        ordered_games = games.copy(deep = True)
        if len(cat.get())>0:
            ordered_games = ft.rating_filter(ordered_games, 'category', cat.get())
        if len(mech.get())>0:
            ordered_games = ft.rating_filter(ordered_games, 'mechanic', mech.get())
        if len(des.get())>0:
            ordered_games = ft.rating_filter(ordered_games, 'designer', des.get())
        if len(pub.get())>0:
            ordered_games = ft.rating_filter(ordered_games, 'publisher', pub.get())
        out = list(ft.select_next_n(ordered_games,9))[0]['name']
        print(out)
        for i in range(len(out)):
            game_objs[i].set(out[i])

            if i==0:
                button0['image'] = game_objs[i].my_img
            elif i==1:
                button1['image'] = game_objs[i].my_img
            elif i==2:
                button2['image'] = game_objs[i].my_img
            elif i==3:
                button3['image'] = game_objs[i].my_img
            elif i==4:
                button4['image'] = game_objs[i].my_img
            elif i==5:
                button5['image'] = game_objs[i].my_img
            elif i==6:
                button6['image'] = game_objs[i].my_img
            elif i==7:
                button7['image'] = game_objs[i].my_img
            elif i==8:
                button8['image'] = game_objs[i].my_img
        default = ImageTk.PhotoImage(Image.open(default_img))
        for i in range(len(out),9):
            game_objs[i].set(' ')
            if i==0:
                button0['image'] = default
            elif i==1:
                button1['image'] = default
            elif i==2:
                button2['image'] = default
            elif i==3:
                button3['image'] = default
            elif i==4:
                button4['image'] = default
            elif i==5:
                button5['image'] = default
            elif i==6:
                button6['image'] = default
            elif i==7:
                button7['image'] = default
            elif i==8:
                button8['image'] = default

    else:
        print(o.get())
    return

def clear_lists():
    global arg_list, game_weights, excluded
    arg_list = []
    game_weights = []
    excluded = []
    game_table.delete(*game_table.get_children())
    exclude_table.delete(*exclude_table.get_children())

def match_string():
    hits = []
    got = auto.get().lower()
    for item in names:
        if item.lower().startswith(got):
            hits.append(item)
    return hits

def get_typed(event):
    if len(event.keysym) == 1:
        hits = match_string()
        show_hit(hits)

def show_hit(lst):
    if len(lst) >0:
        auto.set(lst[0])
        detect_pressed.filled = True

def detect_pressed(event):
    key = event.keysym
    if len(key) == 1 and detect_pressed.filled is True:
        pos = autofill.index(INSERT)
        autofill.delete(pos, END)

detect_pressed.filled = False
mode = 0
arg_list = []
game_weights = []
excluded = []
image_files = [] #list of image files for removal at end of runtime
###################################


root = Tk()
root.title("Board Game Recomender")
#root.iconbitmap(" path of .ico file")

#create elements
search_type_options = ['similar', 'filter', 'user']#add as recommendation systems are made
chosen_type = StringVar()
chosen_type.set(search_type_options[0])
search_type_drop = OptionMenu(root,chosen_type,*search_type_options,command = search_switcher)

similar_frame = Frame(root)
filter_frame = Frame(root)
user_frame = Frame(root)

#similar display
name_label = Label(similar_frame, text = "Game Name")
auto = StringVar()
autofill = Entry(similar_frame, textvariable=auto)
weight_label = Label(similar_frame, text ="Weight")
w = StringVar()
weight = Entry(similar_frame, textvariable = w)

#similar display outside main frame
var1 = IntVar()
exclude = Checkbutton(root, text="Exclude", variable=var1)
game_table_cols = ('Played Games', 'Weight')
game_table = ttk.Treeview(root, columns=game_table_cols, show='headings',height =5)
for col in game_table_cols:
    game_table.heading(col, text=col)
exclude_table = ttk.Treeview(root, columns=('Games',), show='headings', height = 5)
exclude_table.heading('Games', text='Games to Exclude')

#filter display
cat = StringVar()
category_choice = ttk.Combobox(filter_frame, textvariable = cat, values = cat_list)
mech = StringVar()
mechanic_choice = ttk.Combobox(filter_frame, textvariable = mech, values = mech_list)
des = StringVar()
designer_choice = ttk.Combobox(filter_frame, textvariable = des, values = des_list)
pub = StringVar()
publisher_choice = ttk.Combobox(filter_frame, textvariable = pub, values = pub_list)

#user display
user_label = Label(user_frame, text = "User Name")
o = StringVar()
other = Entry(user_frame, textvariable = o)

#controller display
submit = Button(root, text= 'Add',command = search_handeler)
clear = Button(root, text= 'Clear',command = clear_lists)
exitButton =Button(root, text = "Close Recommender", command = root.quit)

#output display
game0 = Game(' ')
button0 = Button(root, image = game0.my_img,command = lambda: openLink(game0.ID) ,height = 180, width = 200)
game1 = Game(' ')
button1 = Button(root, image = game1.my_img,command = lambda: openLink(game1.ID) ,height = 180, width = 200)
game2 = Game(' ')
button2 = Button(root, image = game2.my_img,command = lambda: openLink(game2.ID) ,height = 180, width = 200)
game3 = Game(' ')
button3 = Button(root, image = game3.my_img,command = lambda: openLink(game3.ID) ,height = 180, width = 200)
game4 = Game(' ')
button4 = Button(root, image = game4.my_img,command = lambda: openLink(game4.ID) ,height = 180, width = 200)
game5 = Game(' ')
button5 = Button(root, image = game5.my_img,command = lambda: openLink(game5.ID) ,height = 180, width = 200)
game6 = Game(' ')
button6 = Button(root, image = game6.my_img,command = lambda: openLink(game6.ID) ,height = 180, width = 200)
game7 = Game(' ')
button7 = Button(root, image = game7.my_img,command = lambda: openLink(game7.ID) ,height = 180, width = 200)
game8 = Game(' ')
button8 = Button(root, image = game8.my_img,command = lambda: openLink(game8.ID) ,height = 180, width = 200)
game_objs = [game0, game1, game2, game3, game4, game5, game6, game7, game8]



#add elements to screen
search_type_drop.grid(row = 0, column = 0)

#uncomment whichever is default first screen
similar_frame.grid(row =0, column =1, columnspan = 2, rowspan = 2)
#filter_frame.grid(row =0, column =1, columnspan = 2, rowspan = 2)
#user_frame.grid(row =0, column =1, columnspan = 2, rowspan = 2)

name_label.grid(row = 0, column = 0)
autofill.grid(row = 0, column = 1)
autofill.bind('<KeyRelease>', get_typed)
autofill.bind('<Key>', detect_pressed)
weight_label.grid(row = 1, column =0)
weight.grid(row = 1, column =1)

category_choice.grid(row = 0, column = 0)
mechanic_choice.grid(row = 0, column = 1)
designer_choice.grid(row = 1, column = 0)
publisher_choice.grid(row = 1, column = 1)

user_label.grid(row = 0, column = 1)
other.grid(row = 0, column = 2)

submit.grid(row = 2, column = 0)
clear.grid(row = 2, column = 1)
exitButton.grid(row=2,column = 2 )

exclude.grid(row = 1, column = 0)
game_table.grid(row=3, column=1, columnspan=2, pady=10)
exclude_table.grid(row=3, column=0, columnspan=1, pady=10)

button0.grid(row =4, column = 0)
button1.grid(row =4, column = 1)
button2.grid(row =4, column = 2)
button3.grid(row =5, column = 0)
button4.grid(row =5, column = 1)
button5.grid(row =5, column = 2)
button6.grid(row =6, column = 0)
button7.grid(row =6, column = 1)
button8.grid(row =6, column = 2)




root.mainloop()

for fileName in image_files: #removes all the image files created in runtime
    os.remove(fileName)
