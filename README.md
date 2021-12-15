<p align="center">
  <a href="data/nhl/nhl_img.jpg">
    <img src="data/nhl/nhl_img.jpg" alt="NHL" width="1200" height="250">
  </a>
</p>

<hr style="border:1px solid gray"> </hr>

<a name="title"></a>
## **Goal or No Goal?**

- [Project Motivation](#motivation)
- [Installation](#installation)
- [Datasets](#datasets)
- [File Descriptions](#files)
- [How to use the project](#usage)
- [Authors, Licensing, Acknowledgments](#authors)

<hr style="border:1px solid gray"> </hr>

<a name="motivation"></a>
## *Project Motivation*

Hockey is an amazing game. There are several datasets providing very detailed information on  players and games. You can find very impressive statistics for particular players, explore their salary, get detailed event recordings for many games. This project is providing data analysis based on those datasets. I partular, the goals of the project are:

- Exloring and joining data from several sources and datasets. 
- Extracting interesting facts, visualizing data in different plots, exploring data needed for later prediction.
- **Predicting probability of a goal based on shot circumstances.** 
- Select and define proper metrics to evaluate the predicting model.
- Improve model according to selected metrics.

In order to achieve those goals, two dataset from two different sources have been used:

- [NHL Player Salaries](https://www.kaggle.com/camnugent/predict-nhl-player-salaries) (player statistics, incl. salaries)
- [NHL Game Data](https://www.kaggle.com/martinellis/nhl-game-data) (Game, team, player and plays information including x,y coordinates)



<a name="installation"></a>
## *Installation*
To open and execute all jupyter notebook in this project, install the newest [Anaconda distribution](https://docs.anaconda.com/anaconda/install/). Following python libraries have been used:
- *pandas (1.2.4)*
- *numpy (1.20.1)*
- *maplotlib (3.3.4)*
- *folium (0.12.1)*
- *seaborn (0.11.1)*
- *sklearn (0.24.1)*
- *lightGBM (3.2.1)*. 

In case you are missing any of the libs after you installed Anaconda, use following syntax on the command line to install:
 
 ```
pip3 install <name_of_the_lib> 
```
Python version used for the development was *Python 3.8*. 
<br>

<a name="datasets"></a>
## *Datasets*
It was not possible to push all dataset to GitHub dua to a large size. In order to re-execute the project, please download dataset into following folders:

- [NHL Player Salaries](https://www.kaggle.com/camnugent/predict-nhl-player-salaries) to folder [data/nhl/nhl_salaries/](data/nhl/nhl_salaries/)
- [NHL Game Data](https://www.kaggle.com/martinellis/nhl-game-data) for folder [data/nhl/nhl_stats/](data/nhl/nhl_stats/)

<br>

<a name="files"></a>
## *File Descriptions*
The core of the project are five jupyter notebooks (order in the name) and two python files providing data and linear model related functions. The best way to start is to open and review one jupyter notebook after another (starting with **#1**, finishing with **#5**). The particular notebooks provide following insights:
<br>
<br>

[**1 Explore data.ipynb**](1.%20Explore%20Data.ipynb) - provides first insight into the data. The content of all avaivable files is explored and usability of this content for next investigations is evaluated. This includes checking of data completeness, first simple plots with interesting values. <br>
For listings, this includes information like what districts are the listings distributed over, what are common and pretty uncommon property types to book (yes, **a cave** or **a room in inglo** are bookable as well :-)). Similar basic investigations have been done for *reviews* and *calendar* files.
<br>
<br>
[**2 Best Time To Visit Munich.ipynb**](2.%20Best%20Time%20To%20Visit%20Munich.ipynb) - answers the question, what is the most expensive and the cheapest time of the year to visit Munich. The price distriribution over the year and the week is discovered.
<br>
<br>
[**3. Where to find the best place.ipynb**](3.%20Where%20to%20find%20the%20best%20place.ipynb) - this notebook focuses to the question, what is the best district and property type to book to save some money but still have a good quality (measured by review scores). It analyses and evaluates prices and reviews per districts. Prices of property types per disctrict are evaluated as well. Especialy to see, what are the most and least expensive property types within the district of your choice.
<br>
<br>
[**4. Prepare data.ipynb**](4.%20Prepare%20data.ipynb) - this notebook prepares data for linear regression. The model will predict price of a listing. In order to prepare the data, we need to drop columns or fill missing values, set proper data types, split multi-strings into separated columns, resolve categorical variables, cut-off outliers. The first regression will be executed and evaluated.
<br>
<br>
[**5. Feature engineering.ipynb**](5.%20Feature%20engineering.ipynb) - this notebook picks all results from the previous notebook and performs some feature engineeing steps to improve the results of the linear regression.
<br>
<br>
[**data_utils.py**](data_utils.py) - provides utilities used in the jupyter notebooks above to work with pandas data frames and columns.
<br>
[**model_utils.py**](model_utils.py) - provides utilities to execute the linear regression and to optimize the feature selection.

<a name="usage"></a>
## *How to use the project* 
If you want to re-run all jupyter notebooks in this project, the best way is to create a folder on your local PC and clone the project using:
 ```
git clone https://github.com/rmnng/dsblogpost.git 
```
In the sub-folder *munich* you can find all current dataset files downloaded from [AirBnB](http://insideairbnb.com/munich/). Feel feel to download recent data and re-execute all jupyter notebooks. It's possible you get slightly different results using the newest data though.
<br>

<a name="authors"></a>
## *Authors, Licensing, Acknowledgments*
Author: Roman Nagy
<br>
License: See [license file](LICENSE)
<br>
<br>
Acknowledgment: 
- Thanks AirBnB for the [data](http://insideairbnb.com/munich/).
- Thanks [Udacity](https://www.udacity.com/) for the inspiration.

[top](#motivation)