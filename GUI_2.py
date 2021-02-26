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
reviews = pd.read_csv('Data/bgg-15m-reviews.csv', usecols = ['user', 'rating', 'name'])
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
users = set(reviews['user'].astype(str).values)

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
    reset_tk_vars()
    clear_game_lists()
    clear_recommended_games()
    clear_filters()
    if search_type == 'similar':
        similar_frame.grid(row = 1, column = 0, columnspan = 3, pady=10)
        filter_frame.grid_remove()
        user_frame.grid_remove()
        exclude_table.grid()
        game_table.grid()
    elif search_type == 'user':
        similar_frame.grid_remove()
        filter_frame.grid_remove()
        user_frame.grid(row = 1, column = 0, columnspan = 3, pady=10)
        exclude_table.grid()
        game_table.grid()
    elif search_type == 'filter':
        similar_frame.grid_remove()
        filter_frame.grid(row = 1, column = 0, columnspan = 3, pady=10)
        user_frame.grid_remove()
        exclude_table.grid_remove()
        game_table.grid_remove()

def similar_search_handler():
    name = auto.get()
    weight = w.get()
    game_weights.append(float(weight))
    arg_list.append(name)
    game_table.insert('', 'end', values=(name, weight))

    #call recommender and put in recs list
    print(arg_list)
    print(game_weights)
    print(excluded)

    update_recommended_games()

def user_recommend_handler():
    global arg_list, game_weights
    user_name = user.get()
    game_ratings = recommender.user_game_ratings(reviews, user_name)
    arg_list, ratings_ls = zip(*game_ratings)
    game_weights = recommender.rating_to_weights(ratings_ls)

    for g, w in zip(arg_list, game_weights):
        game_table.insert('', 'end', values = (g, w))
    
    print(arg_list)
    print(game_weights)
    print(excluded)
    
    update_recommended_games()


def filter_handler():
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

    update_game_buttons(out)

def update_recommended_games():
    recs = recommender.recommend_games(recommendation_df, arg_list, similarity_matrices,game_weights=game_weights, similarity_weights=similarity_weights, num_games=9, exclude=excluded)
    update_game_buttons(recs)

def update_game_buttons(names):
    num_games = min(len(names), 9)
    for i in range(num_games):
        game_objs[i].set(names[i])

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
    for i in range(num_games,9):
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

def clear_recommended_games():
    for i in range(9):
        game_objs[i].set(' ')
    default = ImageTk.PhotoImage(Image.open(default_img))
    button0['image'] = default
    button1['image'] = default
    button2['image'] = default
    button3['image'] = default
    button4['image'] = default
    button5['image'] = default
    button6['image'] = default
    button7['image'] = default
    button8['image'] = default

def exclude_handeler():
    name = auto.get()
    excluded.append(name)
    exclude_table.insert('', 'end', values=(name,))
    update_recommended_games()

def clear_game_lists():
    global arg_list, game_weights, excluded
    arg_list = []
    game_weights = []
    excluded = []
    game_table.delete(*game_table.get_children())
    exclude_table.delete(*exclude_table.get_children())

def clear_filters():
    cat.set('')
    mech.set('')
    des.set('')
    pub.set('')
    

def match_string(ls, tk_var):
    hits = []
    got = tk_var.get().lower()
    for item in ls:
        if item.lower().startswith(got):
            hits.append(item)
    return hits

def autofill_get_typed(event):
    if len(event.keysym) == 1:
        hits = match_string(names, auto)
        autofill_show_hit(hits)

def autofill_show_hit(lst):
    if len(lst) >0:
        auto.set(lst[0])
        autofill_detect_pressed.filled = True

def autofill_detect_pressed(event):
    key = event.keysym
    if len(key) == 1 and autofill_detect_pressed.filled is True:
        pos = autofill.index(INSERT)
        autofill.delete(pos, END)

def game_autofill_get_typed(event):
    if len(event.keysym) == 1:
        hits = match_string(names, auto)
        game_autofill_show_hit(hits)

def game_autofill_show_hit(lst):
    if len(lst) >0:
        auto.set(lst[0])
        game_autofill_detect_pressed.filled = True

def game_autofill_detect_pressed(event):
    key = event.keysym
    if len(key) == 1 and game_autofill_detect_pressed.filled is True:
        pos = game_autofill.index(INSERT)
        game_autofill.delete(pos, END)

def user_autofill_get_typed(event):
    if len(event.keysym) == 1:
        hits = match_string(users, user)
        user_autofill_show_hit(hits)

def user_autofill_show_hit(lst):
    if len(lst) >0:
        user.set(lst[0])
        user_autofill_detect_pressed.filled = True

def user_autofill_detect_pressed(event):
    key = event.keysym
    if len(key) == 1 and user_autofill_detect_pressed.filled is True:
        pos = user_autofill.index(INSERT)
        user_autofill.delete(pos, END)

def reset_tk_vars():
    auto.set('')
    w.set('')
    user.set('')

autofill_detect_pressed.filled = False
game_autofill_detect_pressed.filled = False
user_autofill_detect_pressed.filled = False
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
search_type_options = ['similar', 'filter','user']#add as recommendation systems are made
chosen_type = StringVar()
chosen_type.set(search_type_options[0])
search_type_drop = OptionMenu(root,chosen_type,*search_type_options,command = search_switcher)

similar_frame = Frame(root)
filter_frame = Frame(root)
user_frame = Frame(root)


#similar display
auto = StringVar()
w = StringVar()
name_label = Label(similar_frame, text = "Game Name")
autofill = Entry(similar_frame, textvariable=auto)
submit = Button(similar_frame, text= 'Add', command = similar_search_handler)
clear = Button(similar_frame, text= 'Clear', command = clear_game_lists)
exclude = Button(similar_frame, text="Exclude", command = exclude_handeler)
weight_label = Label(similar_frame, text ="Weight")
weight = other = Entry(similar_frame, textvariable = w)

#user recommendation display
user = StringVar()
name_label_user = Label(user_frame, text = "Game Name")
game_autofill = Entry(user_frame, textvariable=auto)
exclude_user = Button(user_frame, text="Exclude", command = exclude_handeler)
user_label = Label(user_frame, text = "User Name")
user_autofill = Entry(user_frame, textvariable=user)
user_recommend = Button(user_frame, text="Recommend", command = user_recommend_handler)

#filter display
cat = StringVar()
mech = StringVar()
des = StringVar()
pub = StringVar()
category_label = Label(filter_frame, text='Category: ')
category_choice = ttk.Combobox(filter_frame, textvariable = cat, values = cat_list)
mechanic_label = Label(filter_frame, text='Mechanic: ')
mechanic_choice = ttk.Combobox(filter_frame, textvariable = mech, values = mech_list)
designer_label = Label(filter_frame, text='Designer: ')
designer_choice = ttk.Combobox(filter_frame, textvariable = des, values = des_list)
publisher_label = Label(filter_frame, text='Publisher: ')
publisher_choice = ttk.Combobox(filter_frame, textvariable = pub, values = pub_list)
filter_button = Button(filter_frame, text='Filter', command=filter_handler)
clear_filter_button = Button(filter_frame, text='Clear', command=clear_filters)


# Elements outside search frames
exitButton =Button(root, text = "Close Recommender", command = root.quit)

game_table_cols = ('Played Games', 'Weight')
game_table = ttk.Treeview(root, columns=game_table_cols, show='headings', height =5)
for col in game_table_cols:
    game_table.heading(col, text=col)

exclude_table = ttk.Treeview(root, columns=('Games',), show='headings', height =5)
exclude_table.heading('Games', text='Games to Exclude')

game0 = Game(' ')
button0 = Button(root, image = game0.my_img,command = lambda: openLink(game0.ID) ,height = 200, width = 200)
game1 = Game(' ')
button1 = Button(root, image = game1.my_img,command = lambda: openLink(game1.ID) ,height = 200, width = 200)
game2 = Game(' ')
button2 = Button(root, image = game2.my_img,command = lambda: openLink(game2.ID) ,height = 200, width = 200)
game3 = Game(' ')
button3 = Button(root, image = game3.my_img,command = lambda: openLink(game3.ID) ,height = 200, width = 200)
game4 = Game(' ')
button4 = Button(root, image = game4.my_img,command = lambda: openLink(game4.ID) ,height = 200, width = 200)
game5 = Game(' ')
button5 = Button(root, image = game5.my_img,command = lambda: openLink(game5.ID) ,height = 200, width = 200)
game6 = Game(' ')
button6 = Button(root, image = game6.my_img,command = lambda: openLink(game6.ID) ,height = 200, width = 200)
game7 = Game(' ')
button7 = Button(root, image = game7.my_img,command = lambda: openLink(game7.ID) ,height = 200, width = 200)
game8 = Game(' ')
button8 = Button(root, image = game8.my_img,command = lambda: openLink(game8.ID) ,height = 200, width = 200)
game_objs = [game0, game1, game2, game3, game4, game5, game6, game7, game8]





#add default Root elements
search_type_drop.grid(row = 0, column = 0)

similar_frame.grid(row = 1, column = 0, columnspan = 3, pady=10)
# filter_frame.grid(row = 1, column = 0, columnspan = 3, pady=10)
# user_frame.grid(row = 1, column = 0, columnspan = 3, pady=10)

exitButton.grid(row=0,column = 2 )

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

#add similarity_frame elements
name_label.grid(row = 0, column = 1)

autofill.grid(row = 0, column = 2)
autofill.bind('<KeyRelease>', autofill_get_typed)
autofill.bind('<Key>', autofill_detect_pressed)

weight_label.grid(row = 1, column =1)
weight.grid(row = 1, column =2)
submit.grid(row = 0, column = 3)
exclude.grid(row = 0, column = 4)
clear.grid(row = 2, column = 3)

#add user_frame elements
name_label_user.grid(row=0, column=1)
game_autofill.grid(row=0, column=2)
game_autofill.bind('<KeyRelease>', game_autofill_get_typed)
game_autofill.bind('<Key>', game_autofill_detect_pressed)
exclude_user.grid(row=0,column=3)

user_label.grid(row=2, column=1)
user_autofill.grid(row=2, column=2)
user_autofill.bind('<KeyRelease>', user_autofill_get_typed)
user_autofill.bind('<Key>', user_autofill_detect_pressed)
user_recommend.grid(row=2, column=3)

#add filter_frame elements
category_label.grid(row = 0, column = 0)
category_choice.grid(row = 0, column = 1)
mechanic_label.grid(row=0, column=2)
mechanic_choice.grid(row = 0, column = 3)
designer_label.grid(row = 1, column = 0)
designer_choice.grid(row = 1, column = 1)
publisher_label.grid(row = 1, column = 2)
publisher_choice.grid(row = 1, column = 3)
filter_button.grid(row = 2, column = 2)
clear_filter_button.grid(row = 2, column = 3)


root.mainloop()

for fileName in image_files: #removes all the image files created in runtime
    os.remove(fileName)




'''
my_img_file = wget.download("https://cf.geekdo-images.com/thumb/img/HEKrtpTC1y1amXh5cKnVvowyE5Y=/fit-in/200x150/pic1534148.jpg")
#use the thumbnail link for this or the image is way too big
print(' ')
image_files = [] #list of image files for removal at end of runtime
image_files.append(my_img_file)
my_img = ImageTk.PhotoImage(Image.open(my_img_file))


def aFunc():
    print(myInputField.get())
    return

def openLink(gameID):
    print("image pressed")
    url ='https://boardgamegeek.com/boardgame/'+gameID
    webbrowser.open(url)  # Go to example.com
    return


#create
myLabel = Label(root, text ='Hello World!')
myButton = Button(root, text ='I\'m a button', command = aFunc) #set command to function which is called when button is pressed
                                                                #set color with fg or bg for forground or background
                                                                #change size with padx and pady
                                                                #...
exitButton =Button(root, text = "Close", command = root.quit)
myInputField = Entry(root)  #set size with width and height
                            #set color with bg and fg
                            #call object .get to get the contents of input field
                            #call object .insert to set text in the field
                            #...
dispImage = Button(root, image = my_img, command = lambda: openLink('30549'), height = 200, width = 200) #change this to an actual link

options =["1","2","3","4","5"]
clicked = StringVar()
clicked.set(options[0])
#use .get() to extract selected value
drop =OptionMenu(root,clicked,*options)
drop.grid(row =3, column = 0)

#add to screen
myLabel.grid(row =0, column = 0)
myButton.grid(row = 1, column = 1)
myInputField.grid(row =2,column = 1)
dispImage.grid(row = 0, column= 2)
exitButton.grid(row=2,column = 2 )
root.mainloop()





for fileName in image_files: #removes all the image files created in runtime
    os.remove(fileName)
'''
