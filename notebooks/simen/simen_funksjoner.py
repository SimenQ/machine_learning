from shapely.geometry import Point
import pandas as pd
stores_train = pd.read_csv('../../data/raw/stores_train.csv')

def average_revenue_of_chain(dataset_stores):
    "Average revenue of chains in datasett"
    dataset_stores = dataset_stores[(dataset_stores["year"] == 2016)]
    return dataset_stores.groupby(['chain_name'])['revenue'].mean()


def average_revenue_of_mall(dataset_stores):
    "Average revenue of malls in dataset"
    dataset_stores = dataset_stores[(dataset_stores["year"] == 2016)]
    return dataset_stores.groupby(['mall_name'])['revenue'].mean()


def population(dataset_age):
    age_df = dataset_age[(dataset_age["year"] == 2016)]
    population = age_df.drop(["grunnkrets_id", "year"], axis=1).sum(axis=1)
    age_df["population_count"] = population
    return age_df[["grunnkrets_id", "population_count"]]


def population_grouped(data_age, data_geography, grouping_element):
    age_df = population(data_age)
    geography_df = data_geography[data_geography["year"] == 2016]
    population_df = age_df.merge(geography_df, how="left", on="grunnkrets_id")
    grouped_df = population_df.groupby([grouping_element], as_index=False)[
        "population_count"].sum()
    return grouped_df


def mean_income_per_capita(dataset_age, dataset_income):
    "mean income per capita per grunnkrets"
    age_df = population(dataset_age)
    income_df = dataset_income[dataset_income["year"] == 2016]
    age_and_income_df = age_df.merge(income_df, how='left', on='grunnkrets_id')
    mean_income = age_and_income_df.drop(['year', 'singles', 'couple_without_children',
                                         'couple_with_children', 'other_households', 'single_parent_with_children'], axis=1)
    mean_income['mean_income'] = mean_income['all_households'] / \
        mean_income['population_count']
    mean_income = mean_income.drop(['all_households'], axis=1)

    return mean_income


def mean_income_per_capita_grouped(dataset_age, dataset_income, dataset_geography, grouping_element):
    # gets data from mean_income_per_capita functino
    data_mean_income = mean_income_per_capita(dataset_age, dataset_income)
    # gets data from geography set and makes sure we only use data for 2016
    geography_df = dataset_geography[dataset_geography["year"] == 2016]
    # gets the data of mean income with the geography data
    mean_income_geo_df = data_mean_income.merge(
        geography_df, how='left', on='grunnkrets_id')
    # sum the number of people based on grouping element
    grouped_population_df = mean_income_geo_df.groupby(
        [grouping_element], as_index=False)["population_count"].sum()
    # merge this with the grunnkrets to see both total population per selected area and grunnkrets
    total_grouped_df = mean_income_geo_df.merge(
        grouped_population_df, how='left', on=grouping_element)
    portion_income_df = total_grouped_df
    # find ration of grunnkrets to total population and multiply this with grunnkrets mean income
    portion_income_df['mean_income'] = total_grouped_df['mean_income'] * \
        total_grouped_df['population_count_x'] / \
        total_grouped_df['population_count_y']
    # add these incomes together, should add up to the total mean income for the selected area
    grouped_income_df = portion_income_df.groupby(
        [grouping_element], as_index=False)["mean_income"].sum()
    return grouped_income_df

def make_point(stores_train):
    stores_train_new = stores_train.copy()
    stores_train_new['point'] = [Point(xy) for xy in zip(stores_train.lon, stores_train.lat)] 
    stores_train_new_finished = stores_train_new[['store_id','point']]
    return stores_train_new_finished
    



