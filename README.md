
# RaceFit

The main goal of the RaceFit is to help professional cycling team coaches to allocate cyclist to races. Additional goal is to model the decision making process of the coach and understand the most important factors in the decision. I invite you to read the paper "Modelling Coach Decisions in Professional Cycling Teams", describing the method more deeply.

You can find the poster and the presentation presented in the ECML-PKDD 2022 conference in the [*deliverables*](https://github.com/MaorSagi/RaceFit/tree/master/deliverables) directory.

Also, the paper is available [here](https://dtai.cs.kuleuven.be/events/MLSA22/papers/MLSA22_paper_9312.pdf) and the poster [here](https://drive.google.com/file/d/1DBAcgUwpGI6oHocyJHQfuSA4MrQfk17D/view?usp=share_link).

## Installation
It required to have a python environment containing the necessary packages. For you convenience, a file named 'requirements.txt' is attached from which you can install the libraries easily.

In order to install the environment from the file run the following command:

```bash
  pip install -r requirements.txt
```
    
## Data

The database of the cyclists, races, and teams can not be publish currently, part of the data might be publish soon. In the meantime, data description can be found in the paper under "The IPT’s Cyclists’ Workouts and Races Dataset" section.

## Running instructions

These are the actions (tasks) the system allows:

- Create teams cyclist participation in race and in stage
- Create examples and labels input from raw data
- Preprocessing 
- Evaluate popularity baselines
- Training and evaluation of RaceFit

In the next sections I will describe the parameters for each one of the tasks with actual usage examples.


### Create teams cyclist participation in race and in stage

Creating binary matrices of cyclist-race and cyclist-stage. The matrices fill in 1 if the cyclist participated in a race (or stage) and 0 otherwise. 
The team pcs IDs:
| Team |  PCS Id  |
|:-----|:--------:|
| AG2R Citroën Team   | 1060 | 
| CCC Team   | 1103 | 
| Lotto Soudal   | 2088 | 
| Team Jumbo-Visma   | 1330 | 
| Israel - Premier Tech   | 2738 | 
| Cofidis   | 1136 | 
| Groupama - FDJ   | 1187 | 
| Movistar Team   | 2040 | 
| UAE Team Emirates   | 1253 | 
| Trek - Segafredo   | 1258 | 
 
mandatory parameters:
```bash
-a create_matrix
-ti <team pcs id>
```

optional:
```bash
-o 1 (overwrite the existing files)
```


Usage example
```bash
python -a create_matrix -ti 1258 -o 1
```

### Create examples and labels input from raw data

Given race, cyclist and workouts of the last weeks prior the race (we defined it as workouts time window), this process is creating examples. Each example consist of race, cyclist and summarized workout vector, the label of the example is represented by the cyclist participation in race/stage.
It is possible to choose to create the input for the model by stages or by races, the training section will describe this matter.

Possible parameters:
- Imputation: without, SimpleImputer, KNNImputer, IterativeImputer
- Time Window Size: int, number of weeks
- Data Source: STRAVA, TP
- Workouts Aggregation Function: SmartAgg (use both AVG and SUM), Average

mandatory parameters:
```bash
-a create_input
-ti <team pcs id>            # continue the last process of creating matrices by insert the same team id
-iw <workouts imputer>       # choose whether to use imputation method for the workouts table
-t <time window size> 
-ws <data source> 
-af <aggregation function>
```
optional:
```bash
-rp 1                        # change to race prediction instead of stage prediction - which is the default
-o 1                         # overwrite the existing files
```

Usage example
```bash
python -a create_input -ti 2738 -iw without -t 5 -ws STRAVA -af SmartAgg -rp 1 -o 1
```

### Preprocessing

Data cleaning and preprocessing using multiple methods such as drop high-value missing ratio examples features, encoding categorical features, scaling data and data imputation.

Possible parameters:
- Imputation: without, SimpleImputer, KNNImputer, IterativeImputer
- Time Window Size: int, number of weeks
- Data Source: STRAVA, TP
- Workouts Aggregation Function: SmartAgg (use both AVG and SUM), Average
- Examples Features Non-Missing Ratio: without, float (0.4 value will cause dropping examples features with missing ratio of 60% or greater)
- Standardization: StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler

mandatory parameters:
```bash
-a preprocessing
-iw <workouts imputer>                      # continue the last process of creating input by insert the same imputer
-t <time window size>                       # continue the last process of creating input by insert the same number of weeks
-ti <team pcs id>                           # continue the last process of creating input by insert the same team id
-ws <data source>                           # continue the last process of creating input by insert the same data source
-af <aggregation function>                  # continue the last process of creating input by insert the same function
-i <examples imputer>                       # choose whether to use imputation method for the examples
-c <examples features non-missing ratio>
```
optional:
```bash
-rp 1                                       # continue the last process of creating input by insert the same rp value
-o 1                                        # overwrite the existing files
-s <scaler>
```

Usage example
```bash
python -a preprocessing -iw without -t 5 -ti 2738 -ws STRAVA -af SmartAgg  -i SimpleImputer -c 0.4 -o 1 -rp 1 -s StandardScaler
```

### Evaluate popularity baselines

This task is for evaluating the popularity baslines. The popularity values computed as features in the examples.

Possible parameters:
- Imputation: without, SimpleImputer, KNNImputer, IterativeImputer
- Time Window Size: int, number of weeks
- Data Source: STRAVA, TP
- Workouts Aggregation Function: SmartAgg (use both AVG and SUM), Average
- Examples Features Non-Missing Ratio: without, float (0.4 value will cause dropping examples features with missing ratio of 60% or greater)
- Standardization: StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler

mandatory parameters:
```bash
-a eval_baselines
-iw <workouts imputer>                      # continue the last process of preprocessing by insert the same workouts imputer
-t <time window size>                       # continue the last process of preprocessing by insert the same number of weeks
-ti <team pcs id>                           # continue the last process of preprocessing by insert the same team id
-ws <data source>                           # continue the last process of preprocessing by insert the same data source
-af <aggregation function>                  # continue the last process of preprocessing by insert the same function
-i <examples imputer>                       # continue the last process of preprocessing by insert the same imputer
-c <examples features non-missing ratio>    # continue the last process of preprocessing by insert the same c value
```
optional:
```bash
-rp 1                                       # continue the last process of preprocessing by insert the same rp value
-o 1                                        # overwrite the existing files
-s <scaler>                                 # continue the last process of preprocessing by insert the same scaler
-oi 1                                       # evaluate only important races (taken from PCS dropdown races list)
```

Usage example
```bash
python -a eval_baselines -iw without -i SimpleImputer -t 5 -ti 2738 -o 1 -ws STRAVA -af SmartAgg -c 0.4 -oi 1
```

### Train and Evaluate RaceFit

The algorithm of RaceFit and its evaluation including training the classifier the algorithm use.

Possible parameters:
- Action: train_model, eval_model or train_eval that combines both
- Imputation: without, SimpleImputer, KNNImputer, IterativeImputer
- Time Window Size: int, number of weeks
- Data Source: STRAVA, TP
- Workouts Aggregation Function: SmartAgg (use both AVG and SUM), Average
- Examples Features Non-Missing Ratio: without, float (0.4 value will cause dropping examples features with missing ratio of 60% or greater)
- Standardization: StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler
- Classifier: CatBoost, AdaBoost, Logistic, DecisionTree, RandomForest, KNN, SVC, XGBoost, LGBM, GaussianNB, GradientBoosting

mandatory parameters:
```bash
-a <action>
-iw <workouts imputer>                      # continue the last process of preprocessing by insert the same workouts imputer
-t <time window size>                       # continue the last process of preprocessing by insert the same number of weeks
-ti <team pcs id>                           # continue the last process of preprocessing by insert the same team id
-ws <data source>                           # continue the last process of preprocessing by insert the same data source
-af <aggregation function>                  # continue the last process of preprocessing by insert the same function
-i <examples imputer>                       # continue the last process of preprocessing by insert the same imputer
-c <examples features non-missing ratio>    # continue the last process of preprocessing by insert the same c value
-m <classifier>
```
optional:
```bash
-rp 1                                       # continue the last process of preprocessing by insert the same rp value
-o 1                                        # overwrite the existing files
-s <scaler>                                 # continue the last process of preprocessing by insert the same scaler
-oi 1                                       # evaluate only important races (taken from PCS dropdown races list)
```

Usage example
```bash
python -a train_eval -iw without -i SimpleImputer -t 5 -ti 2738 -o 1 -ws STRAVA -af SmartAgg -c 0.4 -oi 1 -m CatBoost
```

## Plot Results

The analysis tools I used and are the following:
- Precision@i , Recall@i while i is the number of cyclists recommended (for each parameter while the other parameters results are averaged)
- Recall@(n+k) while n is the # cyclists participated and k is gap for the coach to choose from
- Plot Feature Importance bar plots
- Generate feature importance csv files sorted by ranking of importance for all teams
- Plot Decision Tree top nodes
- Plot Catboost tree top nodes
- Plot the learning curve of the model by time (Time graph)
- AUC of Precision@i-Recall@i graph (for each parameter while the other parameters results are averaged)
- AUC of Precision@i-Recall@i interaction between 2 parameters based on chronological order of use.


The configuration of the graphs to plot you can adjust using the file "results_consts". 

Main configs:
- SINGLE_RACE_TYPE: ONE_DAY_RACES,MAJOR_TOURS, GRAND_TOURS (show results only for one type of race)
- WORKOUTS_SRC: STRAVA, TP 
- SINGLE_RACE_TYPE: None for generating all teams plots or the name of the team (i.e. "Israel - Premier Tech")
- with_baseline: plot baselines lines
- top_i: while plotting Time graph, define the # cyclists recommended


## Support

For support you can contact me in email maors802@gmail.com or you can reach me at LinkedIn.

[![linkedin](https://img.shields.io/badge/linkedin-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/maorsagi/)



