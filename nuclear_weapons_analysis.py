import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import numpy as np
from scipy.signal import argrelextrema
import math

nk = "Number of nuclear warheads"

CONCAVE_UP = "[1945, 1959) U (1968, 1972) U (1974, 1976) U (1977, 1979) U (1980, 1983) U (1991, 1997) U (1998, 2000) U (2003, 2005) U (2006, 2012) U (2015, 2020)"
CONCAVE_DOWN = "(1959, 1968) U (1972, 1974) U (1976, 1977) U (1979, 1980) U (1983, 1991) U (1997, 1998) U (2000, 2003) U (2005, 2006) U (2012, 2015) U (2020, 2022)"

INCREASING = "[1945, 1967) U (1970, 1986) U (2017, 2021)"
DECREASING = "(1967, 1970) U (1986, 2017) U (2021, 2022]"

def convert_to_intervals(interval_string):
    # Remove parentheses and split the string into intervals
    intervals = interval_string.replace("(", "").replace(")", "").replace("[", "").replace("]", "").split(" U ")

    # Convert each interval into a tuple of integers
    intervals = [tuple(map(int, interval.split(", "))) for interval in intervals]

    return intervals
def load_data(p="nuclear_warhead_stockpiles.csv"):
    df = pd.read_csv(p)

    entities = df['Entity'].unique()
    
    entity_dfs = {}
    
    for entity in entities:
        entity_df = filter_entity(df, entity)
        entity_df = entity_df.sort_values(by="Year")
        entity_dfs[entity] = entity_df
    return entity_dfs

def filter_entity(df, entity):
    return df[df["Entity"] == entity]

def compute_gradient(dfk):
    """This computes the gradient of a row in a dataframe"""
    return np.array(np.gradient(np.array(dfk)))

def first_derivative_test(f, f_p):
    """This function will return the maximums and minimums of a function f given its derivative"""

    
def compute_analysis(entity_dfs):
    """
    This function will compute the full analysis from a given entity dataframe
    First and second derivatives will be generated from the data and maximums, mins, and zeros for each will be given
    """

    analysisd = {}

    for entity in entity_dfs:

        first_derivative = compute_gradient(entity)
        print(first_derivative)
        second_derivative = compute_gradient(first_derivative)

        analysisd[entity]["first"] = first_derivative
        analysisd[entity]["second"] = second_derivative

    return analysisd

def generate_derivative_plots(entity_df):
    """
    This function will generate plots for the first and second derivatives of each entity
    """
    first_derivatives = {}
    for entity in entity_df.keys():
        if entity != "World" and entity != "United States" and entity != "Russia":
            first_derivative = compute_gradient(entity_df[entity][nk])
            print(compute_gradient(first_derivative))
            first_derivatives[entity] = first_derivatives

            plt.scatter(entity_df[entity]["Year"], first_derivative)
    plt.title("First Derivative of " + entity)
    plt.xlabel("Year")
    plt.ylabel("First Derivative")
    plt.savefig("figures/derivatives/lesser_nations_first" + entity + ".png")
    plt.close()

    second_derivatives = {}
    for entity in entity_df.keys():
        if entity != "World" and entity != "United States" and entity != "Russia":
            second_derivative = compute_gradient(first_derivatives[entity])
            print(second_derivative)
            second_derivatives[entity] = second_derivative

            plt.scatter(entity_df[entity]["Year"], second_derivative)
    plt.title("Second Derivative")
    plt.xlabel("Year")
    plt.ylabel("Second Derivative")
    plt.savefig("figures/derivatives/second_derivative.png")
    plt.close()
        
    cold_war_nukes = entity_df["United States"][nk][entity_df["United States"]["Year"] < 1991].add(entity_df["Russia"][nk][entity_df["Russia"]["Year"] < 1991], fill_value=0)
    plt.scatter(range(len(list(cold_war_nukes))), cold_war_nukes)
    plt.title("Cold War Nuclear Warheads Stockpiles")
    plt.xlabel("Year")
    plt.ylabel("Estimated Number of Nuclear Warheads")
    plt.savefig("figures/graphs/cold_war_nuclear_warheads.png")
    plt.close()

    first_derivative_cold_war = compute_gradient(cold_war_nukes)
    plt.scatter(range(len(list(cold_war_nukes))), first_derivative_cold_war)
    plt.title("First Derivative of Cold War Nuclear Warheads Stockpiles")
    plt.xlabel("Year")
    plt.ylabel("First Derivative")
    plt.savefig("figures/derivatives/first_derivative_cold_war_nuclear_warheads.png")
    plt.close()

    second_derivative_cold_war = compute_gradient(first_derivative_cold_war)
    plt.scatter(range(len(list(cold_war_nukes))), second_derivative_cold_war)
    plt.title("Second Derivative of Cold War Nuclear Warheads Stockpiles")
    plt.xlabel("Year")
    plt.ylabel("Second Derivative")
    plt.savefig("figures/derivatives/second_derivative_cold_war_nuclear_warheads.png")
    plt.close()

def generate_raw_data_plot():
    data = load_data()

    out = {}
    for entity in data.keys():
        if entity == "United States" or entity == "Russia":
            plt.scatter(data[entity]["Year"], data[entity][nk], label=entity)
            plt.title(f"Nuclear Warhead Stockpiles of {entity} by Year")
            plt.xlabel("Year")
            plt.ylabel("Estimated Number of Nuclear Warheads")
            plt.axhline(0, color='black')
            plt.legend()
            plt.savefig(f"figures/graphs/nuclear_warheads_by_year_{entity}.png")
            plt.close()

            arr = np.array(data[entity][nk])
            analysis = find_extrema_and_inflection_points(arr)
            out[entity] = {}
            out[entity]['analysis'] = analysis

            grad = np.gradient(arr)
            plt.scatter(data[entity]["Year"], grad, label=f"First Derivative of {entity} Nuclear Weapons Stockpiles")
            plt.axhline(0, color='black')
            plt.title(f"First Derivative of {entity} Nuclear Warheads Stockpiles By Year")
            plt.xlabel("Year")
            plt.ylabel("First Derivative (Warheads)")
            plt.legend()
            plt.savefig(f"figures/derivatives/first_derivative_nuclear_warheads_by_year_{entity}.png")
            plt.close()

            out[entity] = {}
            out[entity]['first_derivative'] = grad

            second_grad = np.gradient(grad)
            out[entity]['second_derivative'] = second_grad
            plt.scatter(data[entity]["Year"], second_grad, label=f"Second Derivative of {entity} Nuclear Weapons Stockpiles")
            plt.axhline(0, color='black')
            plt.title(f"Second Derivative of {entity} Nuclear Warheads Stockpiles By Year")
            plt.xlabel("Year")
            plt.ylabel("Second Derivative (Warheads)")
            plt.legend()
            plt.savefig(f"figures/derivatives/second_derivative_nuclear_warheads_by_year_{entity}.png")
            plt.close()

    plt.scatter(data['United States']['Year'], data['United States'][nk], label="United States")
    plt.scatter(data['Russia']['Year'], data['Russia'][nk], label="Russia")
    plt.title("Nuclear Warhead Stockpiles By Year and Country")
    plt.axhline(0, color='black')
    plt.xlabel("Year")
    plt.ylabel("Estimated Number of Nuclear Warheads")
    plt.legend()
    plt.savefig("figures/graphs/nuclear_warheads_by_year_usa_russia.png")
    plt.close()

    plt.scatter(data['United States']['Year'], out['United States']['first_derivative'], label="United States")
    plt.scatter(data['Russia']['Year'], out['Russia']['first_derivative'], label="Russia")
    plt.title("First Derivative of Nuclear Warhead Stockpiles By Year and Country")
    plt.axhline(0, color='black')
    plt.xlabel("Year")
    plt.ylabel("First Derivative (Warheads)")
    plt.legend()
    plt.savefig("figures/derivatives/first_derivative_nuclear_warheads_by_year_usa_russia.png")
    plt.close()

    plt.scatter(data['United States']['Year'], out['United States']['second_derivative'], label="United States")
    plt.scatter(data['Russia']['Year'], out['Russia']['second_derivative'], label="Russia")
    plt.axhline(0, color='black')
    plt.title("Second Derivative of Nuclear Warhead Stockpiles By Year and Country")
    plt.xlabel("Year")
    plt.ylabel("Second Derivative (Warheads)")
    plt.legend()
    plt.savefig("figures/derivatives/second_derivative_nuclear_warheads_by_year_usa_russia.png")
    plt.close()

    combined = []

    for i, year in enumerate(list(data['United States'][nk])):
        combined.append(year+list(data['Russia'][nk])[i])
    analysis = find_extrema_and_inflection_points(np.array(combined))
    

    plt.scatter(data['World']['Year'], combined, label="United States and Soviet Union (Russia post 1991)")
    plt.axhline(0, color='black')
    plt.title("World Nuclear Warheads Stockpiles By Year")
    plt.xlabel("Year")
    plt.ylabel("Estimated Number of Cold War Nuclear Warheads")
    plt.legend()
    plt.savefig("figures/graphs/nuclear_warheads_by_year_world.png")
    plt.close()

    grad = np.gradient(np.array(combined))
    plt.scatter(data['World']['Year'], grad, label="First Derivative of Cold War Nuclear Weapons Stockpiles")
    plt.axhline(0, color='black')
    plt.title("First Derivative of Cold War Nuclear Warheads Stockpiles By Year")
    plt.xlabel("Year")
    plt.ylabel("First Derivative (Warheads)")
    plt.legend()
    plt.savefig("figures/derivatives/first_derivative_nuclear_warheads_by_year_world.png")
    plt.close()
    #print(find_extrema_and_inflection_points(np.array(combined)))

    second_grad = np.gradient(grad)
    plt.scatter(data['World']['Year'], second_grad, label="Second Derivative of Cold War Nuclear Weapons Stockpiles")
    plt.axhline(0, color='black')
    plt.title("Second Derivative of Cold War Nuclear Warheads Stockpiles By Year")
    plt.xlabel("Year")
    plt.ylabel("Second Derivative (Warheads)")
    plt.legend()
    plt.savefig("figures/derivatives/second_derivative_nuclear_warheads_by_year_world.png")
    plt.close()
    
def find_extrema_and_inflection_points(data, add=True):
    maxima = argrelextrema(data, np.greater)[0]
    minima = argrelextrema(data, np.less)[0]

    # Find inflection points
    # An inflection point is where the second derivative changes sign
    second_derivative = np.gradient(np.gradient(data))
    inflection_points = np.where(np.diff(np.sign(second_derivative)))[0]

    _maxima = []
    _minima = []
    _inflection_points = []

    if add:
        for i in maxima:
            _maxima.append((i+1945, data[i]))
        for i in minima:
            _minima.append((i+1945, data[i]))
        for i in inflection_points:
            _inflection_points.append((i+1945, data[i]))
    else:
        for i in maxima:
            _maxima.append((i, data[i]))
        for i in minima:
            _minima.append((i, data[i]))
        for i in inflection_points:
            _inflection_points.append((i, data[i]))
    

    return {"maxima": _maxima, "minima": _minima, "inflection_points": _inflection_points}

def concavity_intervals(data):
    arr = np.array(data)
    analysis = find_extrema_and_inflection_points(arr, add=False)
    inflection_points = analysis['inflection_points']
    inflection_years = [point[0] for point in inflection_points]
    inflection_years.sort()

    # Calculate second derivative
    second_derivative = np.diff(arr, 2)

    positive_intervals = []
    negative_intervals = []

    for i in range(len(inflection_years) - 1):
        # Check the sign of the second derivative in the middle of the interval
        middle = (inflection_years[i] + inflection_years[i+1]) // 2
        if second_derivative[middle] > 0:
            positive_intervals.append(f'({inflection_years[i]+1945}, {inflection_years[i+1]+1945})')
        else:
            negative_intervals.append(f'({inflection_years[i]+1945}, {inflection_years[i+1]+1945})')

    # Add the intervals from the first and last inflection points to negative and positive infinity
    if second_derivative[0] > 0:
        positive_intervals.insert(0, f'(1945, {inflection_years[0]+1945})')
    else:
        negative_intervals.insert(0, f'(1945, {inflection_years[0]+1945})')

    if second_derivative[-1] > 0:
        positive_intervals.append(f'({inflection_years[-1]+1945}, 2022)')
    else:
        negative_intervals.append(f'({inflection_years[-1]+1945}, 2022)')

    # Convert lists to strings in interval union notation
    positive_intervals = ' U '.join(positive_intervals)
    negative_intervals = ' U '.join(negative_intervals)

    return positive_intervals, negative_intervals

def generate_raw_data_plot_line():
    data = load_data()

    out = {}
    for entity in data.keys():
        if entity == "United States" or entity == "Russia":
            plt.plot(data[entity]["Year"], data[entity][nk], label=entity)
            plt.title(f"Nuclear Warhead Stockpiles of {entity} by Year")
            plt.xlabel("Year")
            plt.ylabel("Estimated Number of Nuclear Warheads")
            plt.axhline(0, color='black')
            plt.legend()
            plt.savefig(f"figures/graphs/nuclear_warheads_by_year_{entity}_line.png")
            plt.close()

            arr = np.array(data[entity][nk])
            analysis = find_extrema_and_inflection_points(arr)
            #print(entity, analysis)
            

            grad = np.gradient(arr)
            plt.plot(data[entity]["Year"], grad, label=f"First Derivative of {entity} Nuclear Weapons Stockpiles")
            plt.axhline(0, color='black')
            plt.title(f"First Derivative of {entity} Nuclear Warheads Stockpiles By Year")
            plt.xlabel("Year")
            plt.ylabel("First Derivative (Warheads)")
            plt.legend()
            plt.savefig(f"figures/derivatives/first_derivative_nuclear_warheads_by_year_{entity}_line.png")
            plt.close()

            out[entity] = {}
            out[entity]['first_derivative'] = grad

            second_grad = np.gradient(grad)
            out[entity]['second_derivative'] = second_grad
            plt.plot(data[entity]["Year"], second_grad, label=f"Second Derivative of {entity} Nuclear Weapons Stockpiles")
            plt.axhline(0, color='black')
            plt.title(f"Second Derivative of {entity} Nuclear Warheads Stockpiles By Year")
            plt.xlabel("Year")
            plt.ylabel("Second Derivative (Warheads)")
            plt.legend()
            plt.savefig(f"figures/derivatives/second_derivative_nuclear_warheads_by_year_{entity}_line.png")
            plt.close()

    plt.plot(data['United States']['Year'], data['United States'][nk], label="United States")
    plt.plot(data['Russia']['Year'], data['Russia'][nk], label="Russia")
    plt.title("Nuclear Warhead Stockpiles By Year and Country")
    plt.axhline(0, color='black')
    plt.xlabel("Year")
    plt.ylabel("Estimated Number of Nuclear Warheads")
    plt.legend()
    plt.savefig("figures/graphs/nuclear_warheads_by_year_usa_russia_line.png")
    plt.close()

    plt.plot(data['United States']['Year'], out['United States']['first_derivative'], label="United States")
    plt.plot(data['Russia']['Year'], out['Russia']['first_derivative'], label="Russia")
    plt.title("First Derivative of Nuclear Warhead Stockpiles By Year and Country")
    plt.axhline(0, color='black')
    plt.xlabel("Year")
    plt.ylabel("First Derivative (Warheads)")
    plt.legend()
    plt.savefig("figures/derivatives/first_derivative_nuclear_warheads_by_year_usa_russia_line.png")
    plt.close()

    plt.plot(data['United States']['Year'], out['United States']['second_derivative'], label="United States")
    plt.plot(data['Russia']['Year'], out['Russia']['second_derivative'], label="Russia")
    plt.axhline(0, color='black')
    plt.title("Second Derivative of Nuclear Warhead Stockpiles By Year and Country")
    plt.xlabel("Year")
    plt.ylabel("Second Derivative (Warheads)")
    plt.legend()
    plt.savefig("figures/derivatives/second_derivative_nuclear_warheads_by_year_usa_russia_line.png")
    plt.close()

    combined = [usa + russia for usa, russia in zip(list(data['United States'][nk]), list(data['Russia'][nk]))]

    analysis = find_extrema_and_inflection_points(np.array(combined))
    print(analysis['minima'])
    print(analysis['maxima'])
    print()
    print(analysis['inflection_points'])
    print()
    intervals = concavity_intervals(combined)
    print(intervals[0])
    print()
    print(intervals[1])  

    plt.plot(data['World']['Year'], combined, label="United States and Soviet Union (Russia post 1991)")
    plt.axhline(0, color='black')
    plt.title("World Nuclear Warheads Stockpiles By Year")
    plt.xlabel("Year")
    plt.ylabel("Estimated Number of Cold War Nuclear Warheads")
    plt.legend()
    plt.savefig("figures/graphs/nuclear_warheads_by_year_world_line.png")
    plt.close()

    grad = np.gradient(np.array(combined))
    plt.plot(data['World']['Year'], grad, label="First Derivative of Cold War Nuclear Weapons Stockpiles")
    plt.axhline(0, color='black')
    plt.title("First Derivative of Cold War Nuclear Warheads Stockpiles By Year")
    plt.xlabel("Year")
    plt.ylabel("First Derivative (Warheads)")
    plt.legend()
    plt.savefig("figures/derivatives/first_derivative_nuclear_warheads_by_year_world_line.png")
    plt.close()
    #print(find_extrema_and_inflection_points(np.array(combined)))

    second_grad = np.gradient(grad)
    plt.plot(data['World']['Year'], second_grad, label="Second Derivative of Cold War Nuclear Weapons Stockpiles")
    plt.axhline(0, color='black')
    plt.title("Second Derivative of Cold War Nuclear Warheads Stockpiles By Year")
    plt.xlabel("Year")
    plt.ylabel("Second Derivative (Warheads)")
    plt.legend()
    plt.savefig("figures/derivatives/second_derivative_nuclear_warheads_by_year_world_line.png")
    plt.close()

def generate_ranges_plot():
    data = load_data()
    combined = [usa + russia for usa, russia in zip(list(data['United States'][nk]), list(data['Russia'][nk]))]

    increasing = convert_to_intervals(INCREASING)
    decreasing = convert_to_intervals(DECREASING)
    plt.scatter(data['World']['Year'], combined, label="United States and Soviet Union (Russia post 1991)")
    for interval in increasing:
        plt.axvspan(interval[0], interval[1], color='green', alpha=0.5)
    for interval in decreasing:
        plt.axvspan(interval[0], interval[1], color='red', alpha=0.5)
    plt.axhline(0, color='black')
    plt.title("World Nuclear Warheads Stockpiles By Year")
    plt.xlabel("Year")
    plt.ylabel("Estimated Number of Cold War Nuclear Warheads")
    plt.legend()
    plt.savefig("figures/graphs/nuclear_warheads_by_year_world_increasing_decreasing.png")
    plt.close()

    cup = convert_to_intervals(CONCAVE_UP)
    cdown = convert_to_intervals(CONCAVE_DOWN)

    plt.scatter(data['World']['Year'], combined, label="United States and Soviet Union (Russia post 1991)")
    for interval in cup:
        plt.axvspan(interval[0], interval[1], color='green', alpha=0.5)
    for interval in cdown:
        plt.axvspan(interval[0], interval[1], color='red', alpha=0.5)
    plt.axhline(0, color='black')
    plt.title("World Nuclear Warheads Stockpiles By Year")
    plt.xlabel("Year")
    plt.ylabel("Estimated Number of Cold War Nuclear Warheads")
    plt.legend()
    plt.savefig("figures/graphs/nuclear_warheads_by_year_world_concavity.png")
    plt.close()

def approximator(x):
    if x <= 2010:
        return -1700*abs(x-1986)+64000
    else:
        return 8000
    
def plot_approximator_verus_data():
    data = load_data()
    combined = [usa + russia for usa, russia in zip(list(data['United States'][nk]), list(data['Russia'][nk]))]

    plt.scatter(data['World']['Year'], combined, label="United States and Soviet Union (Russia post 1991)")
    plt.plot(data['World']['Year'], [approximator(x) for x in data['World']['Year']], label="Approximated Function", color="red")
    plt.axhline(0, color='black')
    plt.title("World Nuclear Warheads Stockpiles By Year")
    plt.xlabel("Year")
    plt.ylabel("Estimated Number of Cold War Nuclear Warheads")
    plt.legend()
    plt.savefig("figures/graphs/nuclear_warheads_by_year_world_approximator.png")
    plt.close()

def fit_and_plot_polynomial(degree):
    # Load the data
    data = load_data()

    # Extract the 'World' data and years
    world_data = [usa + russia for usa, russia in zip(list(data['United States'][nk]), list(data['Russia'][nk]))]
    years = list(range(len(world_data)))

    # Fit a polynomial of degree n to the data
    coefficients = np.polyfit(years, world_data, degree)

    # Create a polynomial function from the coefficients
    polynomial = np.poly1d(coefficients)

    # Generate y-values for the polynomial function
    y_values = polynomial(years)

    # Plot the data
    plt.scatter(years, world_data, label="United States and Soviet Union (Russia post 1991)")

    # Plot the polynomial
    plt.plot(years, y_values, label=f"Polynomial of degree {degree}", color="red")

    plt.title("Polynomial Fit of Our Dataset")
    plt.xlabel("Years Since 1945")
    plt.ylabel("Estimated Number of Nuclear Warheads")
    plt.legend()
    plt.savefig(f"figures/graphs/polynomial_fit_degree_{degree}.png")
    plt.close()

    return str(polynomial)


if __name__ == "__main__":
    data = load_data()
    generate_raw_data_plot()
    print("line")
    generate_raw_data_plot_line()
    print("ranges")
    generate_ranges_plot()
    print("approximator")
    plot_approximator_verus_data()
    print("polynomial")
    print(fit_and_plot_polynomial(10))