# Board Game Analysis and Recommendation System
WI2021 ECE143 Project Team1

## Requirements
- Data files should be downloaded from [Kaggle](https://www.kaggle.com/jvanelteren/boardgamegeek-reviews)
	- place both `bgg-15m-reviews.csv` and `games_detailed_info.csv` in `Data/` directory
- `environment.yml` records the modules and their versions used for successfully running
	- Third-party modules: numpy, pandas, nltk, scikit-learn, matplotlib, seaborn, wordcloud, wget, Pillow
	- Create a conda environemnt by running the following command: `conda env create -f environment.yml`

## View Data Analysis
- Open `Main.ipynb` to view/generate all presented data analysis figures and results

## Launch GUI
- Call the following command in the root directory: `python3 gui.py`
- For a full demo on how to interact with GUI, refer to [this video](https://drive.google.com/file/d/1eyYpNlgix89k1wCbbHunqw-vhSToGns_/view?usp=sharing)

## File Descriptions
1. `preprocessing.py`: contains functions used to preprocess data
2. `analysis.py`: contains functions for generating data analysis plots
4. `recommendation_utils.py`: contains functions used to power the recommendation system
5. `filter.py`: contains filtering functions used for the GUI
6. `gui.py`: contains `Game` and `Gui` classes used to build and run the GUI
7. `Main.ipynb`: Jupyter notebook that generates visuals from the analysis and the recommendation system
8. `assignment_test_case.ipynb`: Jupyter notebook that includes assignment test cases for HW1
9. `Data/`: directory that contains games and reviews data used for analysis and recommendation along with a blank image used in `gui.py`
10. `presentation.pdf`: pdf file of the presentation
