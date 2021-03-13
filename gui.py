import os
from tkinter import *
from tkinter import ttk
import webbrowser
import filter as ft
import numbers

import pandas as pd
import numpy as np
import wget
from PIL import ImageTk, Image

import preprocessing as pre
import recommendation_utils as recommender


class Game:
    """Object to link game name to its id and thumbnail"""
    default_img = 'Data/default_image.jpg'

    def __init__(self, name='', id_number=0, thumbnail='', imgfile="Data/default_image.jpg"):
        assert isinstance(name, str)
        assert isinstance(id_number, numbers.Integral)
        assert isinstance(thumbnail, str)
        assert isinstance(imgfile, str)

        self.name = name
        self.id = id_number
        self.thumbnail = thumbnail
        if os.path.exists(imgfile):
            self.img = ImageTk.PhotoImage(Image.open(imgfile))
        else:
            my_img_file = wget.download(self.thumbnail)
            self.img = ImageTk.PhotoImage(Image.open(my_img_file))
            os.remove(my_img_file)

    def set(self, name, id_number, thumbnail, imgfile=None):
        assert isinstance(name, str)
        assert isinstance(id_number, numbers.Integral)
        assert isinstance(thumbnail, str)
        assert imgfile is None or isinstance(imgfile, str)

        self.name = name
        self.id = id_number
        self.thumbnail = thumbnail
        if imgfile is not None and os.path.exists(imgfile):
            self.img = ImageTk.PhotoImage(Image.open(imgfile))
        else:
            my_img_file = wget.download(self.thumbnail)
            self.img = ImageTk.PhotoImage(Image.open(my_img_file))
            os.remove(my_img_file)


class Gui:
    search_type_options = ['similar', 'filter', 'user']
    similarity_cols = ['description', 'category_encoded', 'mechanic_encoded', 'publisher_encoded',
                       'designer_encoded', 'complexity']  # Columns used to find similar games
    similarity_types = ['text', 'one_hot', 'one_hot', 'one_hot', 'one_hot', 'scalar']  # Similarity metric types
    similarity_weights = [0.1, 0.25, 0.25, 0.1, 0.1, 0.2]  # Weights corresdponign to similarity_cols
    n_displayed_games = 9  # Number of games to be displayed in GUI

    def __init__(self, games, reviews):
        assert isinstance(games, pd.DataFrame)
        assert isinstance(reviews, pd.DataFrame)

        # GUI data elements (games and users information)
        self.reviews = reviews
        self.games = games
        self.users = set(reviews['user'].astype(str).values)
        self.names = games['name'].values
        self.ids = games['id'].values
        self.thumbs = games['thumbnail'].values

        # Setup tkinter elements
        self.root = Tk()
        self.auto = StringVar()
        self.w = StringVar()
        self.user = StringVar()
        self.cat = StringVar()
        self.mech = StringVar()
        self.des = StringVar()
        self.pub = StringVar()
        self.arg_list = []
        self.game_weights = []
        self.excluded = []
        self.chosen_type = StringVar()
        self.chosen_type.set(Gui.search_type_options[0])

        # GUI frames
        self.similar_frame = Frame(self.root)
        self.filter_frame = Frame(self.root)
        self.user_frame = Frame(self.root)

        # GUI tables
        self.exclude_table = ttk.Treeview(self.root, columns=('Games',), show='headings', height=5)
        self.exclude_table.heading('Games', text='Games to Exclude')
        game_table_cols = ('Played Games', 'Weight')
        self.game_table = ttk.Treeview(self.root, columns=game_table_cols, show='headings', height=5)
        for col in game_table_cols:
            self.game_table.heading(col, text=col)

        # GUI game buttons and maes
        self.game_objs = [Game() for i in range(Gui.n_displayed_games)]
        self.buttons = [
            Button(self.root, image=game.img, command=lambda: self.open_link(game.id), height=150,
                   width=200)
            for game in self.game_objs]
        self.displayed_names = [Label(self.root, text="") for i in range(len(self.game_objs))]

    def open_link(self, game_id):
        """
        Open the boardgamesgeek link corresponding to the given game_id

        game_id: the id of a game
        """
        assert isinstance(game_id, numbers.Integral) or isinstance(game_id, int)

        url = 'https://boardgamegeek.com/boardgame/' + str(game_id)
        webbrowser.open(url)
        return

    def search_switcher(self, search_type):
        """
        Switch the search type

        search_type: search type of either "similar", "user" or "filter"
        """
        assert isinstance(search_type, str) and search_type in {"similar", "user", "filter"}

        # Clear the current configurations
        self.reset_tk_vars()
        self.clear_game_lists()
        self.clear_recommended_games()
        self.clear_filters()
        if search_type == 'similar':
            self.similar_frame.grid(row=1, column=0, columnspan=3, pady=10)
            self.filter_frame.grid_remove()
            self.user_frame.grid_remove()
            self.exclude_table.grid()
            self.game_table.grid()

        elif search_type == 'user':
            self.similar_frame.grid_remove()
            self.filter_frame.grid_remove()
            self.user_frame.grid(row=1, column=0, columnspan=3, pady=10)
            self.exclude_table.grid()
            self.game_table.grid()

        elif search_type == 'filter':
            self.similar_frame.grid_remove()
            self.filter_frame.grid(row=1, column=0, columnspan=3, pady=10)
            self.user_frame.grid_remove()
            self.exclude_table.grid_remove()
            self.game_table.grid_remove()

    def update_recommended_games(self):
        """ Update the recommended games wit the given configurations and update the game buttons """
        similarity_matrices = recommender.get_similarity_matrices(self.games, Gui.similarity_cols, Gui.similarity_types)
        recs = recommender.recommend_games(self.games, self.arg_list, similarity_matrices, game_weights=self.game_weights,
                                           similarity_weights=Gui.similarity_weights, num_games=9, exclude=self.excluded)
        self.update_game_buttons(recs)

    def user_recommend_handler(self):
        """ Handler for user recommender """
        user_name = self.user.get()
        game_ratings = recommender.user_game_ratings(self.reviews, user_name)
        arg_list, ratings_ls = zip(*game_ratings)
        self.arg_list = arg_list
        self.game_weights = recommender.rating_to_weights(ratings_ls)

        for g, w in zip(arg_list, self.game_weights):
            self.game_table.insert('', 'end', values=(g, w))
        self.update_recommended_games()

    def similar_search_handler(self):
        """ Handler for similar game recommender """
        name = self.auto.get()
        weight = self.w.get()
        self.game_weights.append(float(weight))
        self.arg_list.append(name)
        self.game_table.insert('', 'end', values=(name, weight))
        self.update_recommended_games()

    def filter_handler(self):
        """ Handler for filter """
        ordered_games = self.games.copy(deep=True)
        if len(self.cat.get()) > 0:
            ordered_games = ft.rating_filter(ordered_games, 'category', self.cat.get())
        if len(self.mech.get()) > 0:
            ordered_games = ft.rating_filter(ordered_games, 'mechanic', self.mech.get())
        if len(self.des.get()) > 0:
            ordered_games = ft.rating_filter(ordered_games, 'designer', self.des.get())
        if len(self.pub.get()) > 0:
            ordered_games = ft.rating_filter(ordered_games, 'publisher', self.pub.get())
        out = list(ft.select_next_n(ordered_games, 9))[0]['name']
        self.update_game_buttons(out)

    def exclude_handeler(self):
        """ Handler for inputted games to be excluded """
        name = self.auto.get()
        self.excluded.append(name)
        self.exclude_table.insert('', 'end', values=(name,))
        self.update_recommended_games()

    def clear_game_lists(self):
        """ Clear inputted games """
        self.arg_list = []
        self.game_weights = []
        self.excluded = []
        self.game_table.delete(*self.game_table.get_children())
        self.exclude_table.delete(*self.exclude_table.get_children())

    def clear_filters(self):
        """ Clear filers for categories, mechanics, designers and publishers """
        self.cat.set('')
        self.mech.set('')
        self.des.set('')
        self.pub.set('')

    def reset_tk_vars(self):
        """ Clear tkinter variables """
        self.auto.set('')
        self.w.set('')
        self.user.set('')

    def update_game_buttons(self, names):
        """
        Update GUI buttons with the images and names of provided games

        names: list of game names to be displayed
        """
        assert names is None or (isinstance(names, (np.ndarray, pd.Series, list, tuple)) and all(isinstance(x, str) for x in names))

        # Fill out the buttons with the recommended game images
        num_games = min(len(names), Gui.n_displayed_games)
        for i in range(num_games):
            id_number = self.ids[np.where(self.names == names[i])][0]
            thumbnail = self.thumbs[np.where(self.names == names[i])][0]
            self.game_objs[i].set(names[i], id_number, thumbnail)
            self.buttons[i]["image"] = self.game_objs[i].img
            self.displayed_names[i]["text"] = self.game_objs[i].name

        # Fill out the remaining buttons with the default blank image
        default = ImageTk.PhotoImage(Image.open(Game.default_img))
        for i in range(num_games, Gui.n_displayed_games):
            self.game_objs[i] = Game()
            self.buttons[i]["image"] = default
            self.displayed_names[i]["text"] = " "

    def clear_recommended_games(self):
        """ Clear the recommended game buttons """
        default = ImageTk.PhotoImage(Image.open(Game.default_img))
        for i in range(Gui.n_displayed_games):
            self.game_objs[i] = Game()
            self.buttons[i]["image"] = default
            self.displayed_names[i]["text"] = " "

    def run(self):
        """ Run GUI from end to end """
        def match_string(ls, tk_var):
            """
            Helper function to find an item matching with a given tk variable value

            ls: series/list/tuple of values from which values matching tk_var will be searched
            tk_var: user given tkinter variable
            """
            assert isinstance(ls, (set, pd.Series, np.ndarray, list, tuple))

            hits = []
            got = tk_var.get().lower()
            for item in ls:
                if item.lower().startswith(got):
                    hits.append(item)
            return hits

        def autofill_get_typed(event):
            """ Helper function to suggest games matching with a user inputted game name """
            if len(event.keysym) == 1:
                hits = match_string(self.names, self.auto)
                autofill_show_hit(hits)

        def autofill_show_hit(lst):
            """ Helper function to set the auto with a given list """
            if len(lst) > 0:
                self.auto.set(lst[0])
                autofill_detect_pressed.filled = True

        def autofill_detect_pressed(event):
            """ Helper function to detect whether game autofill was pressed """
            key = event.keysym
            if len(key) == 1 and autofill_detect_pressed.filled is True:
                pos = autofill.index(INSERT)
                autofill.delete(pos, END)

        def game_autofill_get_typed(event):
            """ Helper function to suggest games matching with a user inputted game name """
            if len(event.keysym) == 1:
                hits = match_string(self.names, self.auto)
                game_autofill_show_hit(hits)

        def game_autofill_show_hit(lst):
            """ Helper function to set the auto with a given list """
            if len(lst) > 0:
                self.auto.set(lst[0])
                game_autofill_detect_pressed.filled = True

        def game_autofill_detect_pressed(event):
            """ Helper function to detect whether game autofill was pressed """
            key = event.keysym
            if len(key) == 1 and game_autofill_detect_pressed.filled is True:
                pos = game_autofill.index(INSERT)
                game_autofill.delete(pos, END)

        def user_autofill_get_typed(event):
            """ Helper function to suggest user matching with a user inputted user name """
            if len(event.keysym) == 1:
                hits = match_string(self.users, self.user)
                user_autofill_show_hit(hits)

        def user_autofill_show_hit(lst):
            """ Helper function to set the user auto with a given list """
            if len(lst) > 0:
                self.user.set(lst[0])
                user_autofill_detect_pressed.filled = True

        def user_autofill_detect_pressed(event):
            """ Helper function to detect whether user autofill was pressed """
            key = event.keysym
            if len(key) == 1 and user_autofill_detect_pressed.filled is True:
                pos = user_autofill.index(INSERT)
                user_autofill.delete(pos, END)

        # Initialize the autofill to be "unfilled"
        autofill_detect_pressed.filled = False
        game_autofill_detect_pressed.filled = False
        user_autofill_detect_pressed.filled = False

        # User inputs display: Game
        chosen_type = StringVar()
        chosen_type.set(Gui.search_type_options[0])
        search_type_drop = OptionMenu(self.root, chosen_type, *Gui.search_type_options, command=self.search_switcher)

        name_label = Label(self.similar_frame, text="Game Name")
        autofill = Entry(self.similar_frame, textvariable=self.auto)
        submit = Button(self.similar_frame, text='Add', command=self.similar_search_handler)
        clear = Button(self.similar_frame, text='Clear', command=self.clear_game_lists)
        exclude = Button(self.similar_frame, text="Exclude", command=self.exclude_handeler)
        weight_label = Label(self.similar_frame, text="Weight")
        weight = other = Entry(self.similar_frame, textvariable=self.w)

        # User inputs display: User
        name_label_user = Label(self.user_frame, text="Game Name")
        game_autofill = Entry(self.user_frame, textvariable=self.auto)
        exclude_user = Button(self.user_frame, text="Exclude", command=self.exclude_handeler)
        user_label = Label(self.user_frame, text="User Name")
        user_autofill = Entry(self.user_frame, textvariable=self.user)
        user_recommend = Button(self.user_frame, text="Recommend", command=self.user_recommend_handler)

        # Filter display
        cat_list = list(ft.available_choices(self.games, "category"))
        mech_list = list(ft.available_choices(self.games, "mechanic"))
        des_list = list(ft.available_choices(self.games, "designer"))
        pub_list = list(ft.available_choices(self.games, "publisher"))
        category_label = Label(self.filter_frame, text='Category: ')
        category_choice = ttk.Combobox(self.filter_frame, textvariable=self.cat, values=cat_list)
        mechanic_label = Label(self.filter_frame, text='Mechanic: ')
        mechanic_choice = ttk.Combobox(self.filter_frame, textvariable=self.mech, values=mech_list)
        designer_label = Label(self.filter_frame, text='Designer: ')
        designer_choice = ttk.Combobox(self.filter_frame, textvariable=self.des, values=des_list)
        publisher_label = Label(self.filter_frame, text='Publisher: ')
        publisher_choice = ttk.Combobox(self.filter_frame, textvariable=self.pub, values=pub_list)
        filter_button = Button(self.filter_frame, text='Filter', command=self.filter_handler)
        clear_filter_button = Button(self.filter_frame, text='Clear', command=self.clear_filters)
        exitButton = Button(self.root, text="Close Recommender", command=self.root.quit)

        # Set up grids
        search_type_drop = OptionMenu(self.root, self.chosen_type, *Gui.search_type_options, command=self.search_switcher)
        search_type_drop.grid(row=0, column=0)

        self.similar_frame.grid(row=1, column=0, columnspan=3, pady=10)
        self.game_table.grid(row=3, column=1, columnspan=2, pady=10)
        self.exclude_table.grid(row=3, column=0, columnspan=1, pady=10)

        for i in range(Gui.n_displayed_games):
            self.buttons[i].grid(row=4+2*int(i/3), column=i % 3)
            self.displayed_names[i].grid(row=5+2*int(i/3), column=i % 3)

        name_label.grid(row=0, column=1)

        autofill.grid(row=0, column=2)
        autofill.bind('<KeyRelease>', autofill_get_typed)
        autofill.bind('<Key>', autofill_detect_pressed)

        weight_label.grid(row=1, column=1)
        weight.grid(row=1, column=2)
        submit.grid(row=0, column=3)
        exclude.grid(row=0, column=4)
        clear.grid(row=2, column=3)

        # add user_frame elements
        name_label_user.grid(row=0, column=1)
        game_autofill.grid(row=0, column=2)
        game_autofill.bind('<KeyRelease>', game_autofill_get_typed)
        game_autofill.bind('<Key>', game_autofill_detect_pressed)
        exclude_user.grid(row=0, column=3)

        user_label.grid(row=2, column=1)
        user_autofill.grid(row=2, column=2)
        user_autofill.bind('<KeyRelease>', user_autofill_get_typed)
        user_autofill.bind('<Key>', user_autofill_detect_pressed)
        user_recommend.grid(row=2, column=3)

        # add filter_frame elements
        category_label.grid(row=0, column=0)
        category_choice.grid(row=0, column=1)
        mechanic_label.grid(row=0, column=2)
        mechanic_choice.grid(row=0, column=3)
        designer_label.grid(row=1, column=0)
        designer_choice.grid(row=1, column=1)
        publisher_label.grid(row=1, column=2)
        publisher_choice.grid(row=1, column=3)
        filter_button.grid(row=2, column=2)
        clear_filter_button.grid(row=2, column=3)

        exitButton.grid(row=0, column=2)

        self.root.mainloop()


if __name__ == '__main__':
    games = pd.read_csv("Data/games_detailed_info.csv", index_col=0)  # review stats
    reviews = pd.read_csv('Data/bgg-15m-reviews.csv', usecols=['user', 'rating', 'name'])
    # 1. Remove columns with > 20% of NA values
    key_columns = pre.keep_columns_with_few_na(games)
    readd_columns = ["boardgameexpansion"]
    key_columns.extend(readd_columns)
    # 2. Remove redundant/unnecesary columns
    unnecessary_columns = ["type", "image", "suggested_num_players", "suggested_playerage",
                           "suggested_language_dependence"]
    key_columns = [x for x in key_columns if x not in unnecessary_columns]
    # 3. Rename confusing column names
    games = games.loc[:, key_columns]
    games.rename(columns={"primary": "name", "usersrated": "numratings", "average": "avgrating",
                          "boardgamecategory": "category", "boardgamemechanic": "mechanic",
                          "boardgamedesigner": "designer", "boardgamepublisher": "publisher",
                          "bayesaverage": "bayesavgrating", "Board Game Rank": "rank",
                          "stddev": "stdrating", "median": "medianrating",
                          "owned": "numowned", "trading": "numtrades", "wanting": "numwants",
                          "wishing": "numwishes", "averageweight": "complexity"}, inplace=True)

    # 4. Parse columns with list values
    list_colnames = ["category", "mechanic", "designer", "publisher"]
    games = pre.parse_list_columns(games, list_colnames)
    cat_list = list(ft.available_choices(games, "category"))
    mech_list = list(ft.available_choices(games, "mechanic"))
    des_list = list(ft.available_choices(games, "designer"))
    pub_list = list(ft.available_choices(games, "publisher"))

    # 5. Keep top 10000 games and create encoded columns
    top_10k_games = games[games['rank'] <= 10000].reset_index(drop=True)
    # Encode multi-categorical columns
    recommendation_df = top_10k_games.copy()
    mechanic_encoding = pre.add_encoded_column(recommendation_df, 'mechanic')
    category_encoding = pre.add_encoded_column(recommendation_df, 'category')
    publisher_encoding = pre.add_encoded_column(recommendation_df, 'publisher', 50,
                                                ['(Public Domain)', '(Unknown)', '(Web published)'])
    designer_encoding = pre.add_encoded_column(recommendation_df, 'designer', 20, ['(Uncredited)'])
    # Generate counter columns
    recommender.add_item_counts_column(recommendation_df, 'boardgameexpansion')

    # 6. Create and run a GUI object
    gui = Gui(recommendation_df, reviews)
    gui.run()
