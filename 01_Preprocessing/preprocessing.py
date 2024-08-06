import csv
import shutil
import pandas as pd
from sklearn.linear_model import LinearRegression

class ExpandedList(list):
    def __init__(self, *args):
        super(ExpandedList, self).__init__(args)

    def __sub__(self, other: list):
        return self.__class__(*[item for item in self if item not in other])

ORIGINAL_FILE = 'diabetes_dataset_original.csv'
NEW_FILE = 'diabetes_dataset.csv'
COLUMNS = ExpandedList('Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome')
IGNORED_COLUMNS = ExpandedList('Age', 'BloodPressure')
PRINT_ERRORS = False

def check_line(row: dict, line_number: int) -> int:
    errors = 0

    # Pregnancies
    if 'Pregnancies' not in IGNORED_COLUMNS:
        try:
            if int(row['Pregnancies']) < 0:
                if PRINT_ERRORS: print(f"Error: Row {line_number}, Pregnancies is negative. Value: {row['Pregnancies']}")
                errors += 1
        except ValueError:
            if PRINT_ERRORS: print(f"Error: Row {line_number}, Pregnancies is not a int. Value: {row['Pregnancies']}")
            errors += 1

    # Glucose
    if 'Glucose' not in IGNORED_COLUMNS:
        try:
            if float(row['Glucose']) < 0:
                if PRINT_ERRORS: print(f"Error: Row {line_number}, Glucose is negative. Value: {row['Glucose']}")
                errors += 1
        except ValueError:
            if PRINT_ERRORS: print(f"Error: Row {line_number}, Glucose is not a float. Value: {row['Glucose']}")
            errors += 1

    # BloodPressure
    if 'BloodPressure' not in IGNORED_COLUMNS:
        try:
            if float(row['BloodPressure']) < 0:
                if PRINT_ERRORS: print(f"Error: Row {line_number}, BloodPressure is negative. Value: {row['BloodPressure']}")
                errors += 1
        except ValueError:
            if PRINT_ERRORS: print(f"Error: Row {line_number}, BloodPressure is not a float. Value: {row['BloodPressure']}")
            errors += 1

    # SkinThickness
    if 'SkinThickness' not in IGNORED_COLUMNS:
        try:
            if float(row['SkinThickness']) < 0:
                if PRINT_ERRORS: print(f"Error: Row {line_number}, SkinThickness is negative. Value: {row['SkinThickness']}")
                errors += 1
        except ValueError:
            if PRINT_ERRORS: print(f"Error: Row {line_number}, SkinThickness is not a float. Value: {row['SkinThickness']}")
            errors += 1

    # Insulin
    if 'Insulin' not in IGNORED_COLUMNS:
        try:
            if float(row['Insulin']) < 0:
                if PRINT_ERRORS: print(f"Error: Row {line_number}, Insulin is negative. Value: {row['Insulin']}")
                errors += 1
        except ValueError:
            if PRINT_ERRORS: print(f"Error: Row {line_number}, Insulin is not a float. Value: {row['Insulin']}")
            errors += 1

    # BMI
    if 'BMI' not in IGNORED_COLUMNS:
        try:
            if float(row['BMI']) < 0:
                if PRINT_ERRORS: print(f"Error: Row {line_number}, BMI is negative. Value: {row['BMI']}")
                errors += 1
        except ValueError:
            if PRINT_ERRORS: print(f"Error: Row {line_number}, BMI is not a float. Value: {row['BMI']}")
            errors += 1

    # DiabetesPedigreeFunction
    if 'DiabetesPedigreeFunction' not in IGNORED_COLUMNS:
        try:
            if float(row['DiabetesPedigreeFunction']) < 0:
                if PRINT_ERRORS: print(f"Error: Row {line_number}, DiabetesPedigreeFunction is negative. Value: {row['DiabetesPedigreeFunction']}")
                errors += 1
        except ValueError:
            if PRINT_ERRORS: print(f"Error: Row {line_number}, DiabetesPedigreeFunction is not a float. Value: {row['DiabetesPedigreeFunction']}")
            errors += 1

    # Age
    if 'Age' not in IGNORED_COLUMNS:
        try:
            if int(row['Age']) < 0:
                if PRINT_ERRORS: print(f"Error: Row {line_number}, Age is negative. Value: {row['Age']}")
                errors += 1
        except ValueError:
            if PRINT_ERRORS: print(f"Error: Row {line_number}, Age is not a int. Value: {row['Age']}")
            errors += 1

    return errors

def check_integrity(filename: str) -> int:
    with open(filename, 'r') as file:
        reader = csv.DictReader(file, fieldnames=COLUMNS)
        line_number = 1
        errors = 0
        for row in reader:
            # Skip header
            if line_number == 1:
                line_number += 1
                continue

            errors += check_line(row, line_number)

            line_number += 1

    return errors

def count_rows(filename: str) -> int:
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        return sum(1 for row in reader) - 1

if __name__ == "__main__":
    # Copy diabetes_dataset_original.csv to diabetes_dataset.csv
    print(f'Copying {ORIGINAL_FILE} to {NEW_FILE}')
    shutil.copyfile(ORIGINAL_FILE, NEW_FILE)

    # Check integrity of diabetes_dataset.csv
    print(f'Checking integrity of {NEW_FILE}')
    errors = check_integrity(NEW_FILE)
    print(f"Number of errors: {errors}")
    print(f"Number of rows: {count_rows(NEW_FILE)}")

    # Applying LinearRegression between: BMI and SkinThickness, Glucose and Insulin
    linear_regressions = [['BMI', 'SkinThickness'], ['Glucose', 'Insulin']]
    print("Applying LinearRegression between: BMI and SkinThickness, Glucose and Insulin")
    for regression in linear_regressions:
        df = pd.read_csv(NEW_FILE)
        df = df[df[regression[0]].notnull() & df[regression[1]].notnull()]
        X = df[[regression[0]]]
        y = df[regression[1]]
        model = LinearRegression()
        model.fit(X, y)

        # print(f'Intercept: {model.intercept_}')
        # print(f'Coefficient: {model.coef_[0]}')

        max_a = df[regression[0]].max()
        min_a = df[regression[0]].min()
        with open(NEW_FILE, 'r') as file:
            reader = csv.DictReader(file, fieldnames=COLUMNS)
            rows = []
            line_number = 1
            for row in reader:
                # Skip header
                if line_number == 1:
                    line_number += 1
                    continue

                if row[regression[1]] == '' and row[regression[0]] != '':
                    new_data = pd.DataFrame({regression[0]: [float(row[regression[0]])]})
                    prediction = model.predict(new_data)[0]
                    row[regression[1]] = f"{prediction:.3f}"

                rows.append(row)
                line_number += 1

        with open(NEW_FILE, 'w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=COLUMNS)
            writer.writeheader()
            writer.writerows(rows)

    print("LinearRegression applied successfully")

    # Check integrity of diabetes_dataset.csv
    print(f'Checking integrity of {NEW_FILE}')
    errors = check_integrity(NEW_FILE)
    print(f"Number of errors: {errors}")
    print(f"Number of rows: {count_rows(NEW_FILE)}")

    # Removing lines if contains errors
    print("Removing lines if contains errors")
    with open(NEW_FILE, 'r') as file:
        reader = csv.DictReader(file, fieldnames=COLUMNS)
        rows = []
        line_number = 1
        for row in reader:
            # Skip header
            if line_number == 1:
                line_number += 1
                continue

            if check_line(row, line_number) == 0:
                rows.append(row)

            line_number += 1

    with open(NEW_FILE, 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    # Check integrity of diabetes_dataset.csv
    print(f"Checking integrity of {NEW_FILE}")
    errors = check_integrity(NEW_FILE)
    print(f"Number of errors: {errors}")
    print(f"Number of rows: {count_rows(NEW_FILE)}")

    # Execute operations to change diabetes_dataset.csv
    print("Executing operations to change diabetes_dataset.csv:")
    print("1. Fix each DiabetesPedigreeFunction to contain 3 decimal places")
    print("2. Convert Glucose, BloodPressure, SkinThickness and Insulin to int")
    with open(NEW_FILE, 'r') as file:
        reader = csv.DictReader(file, fieldnames=COLUMNS)
        rows = []
        line_number = 1
        for row in reader:
            # Skip header
            if line_number == 1:
                line_number += 1
                continue

            # Fix each DiabetesPedigreeFunction to contain 3 decimal places
            if 'DiabetesPedigreeFunction' not in IGNORED_COLUMNS:
                row['DiabetesPedigreeFunction'] = f"{float(row['DiabetesPedigreeFunction']):.3f}"

            # Convert Glucose, BloodPressure, SkinThickness and Insulin to int
            if 'Glucose' not in IGNORED_COLUMNS:
                row['Glucose'] = int(float(row['Glucose']))
            if 'BloodPressure' not in IGNORED_COLUMNS:
                row['BloodPressure'] = int(float(row['BloodPressure']))
            if 'SkinThickness' not in IGNORED_COLUMNS:
                row['SkinThickness'] = int(float(row['SkinThickness']))
            if 'Insulin' not in IGNORED_COLUMNS:
                row['Insulin'] = int(float(row['Insulin']))

            rows.append(row)
            line_number += 1

    with open(NEW_FILE, 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    print("Operations executed successfully")

    # Check integrity of diabetes_dataset.csv
    print(f"Checking integrity of {NEW_FILE}")
    check_integrity(NEW_FILE)
    print(f"Number of errors: {errors}")
    print(f"Number of rows: {count_rows(NEW_FILE)}")

    # Remove outliers
    columns = ExpandedList('Age', 'Pregnancies', 'DiabetesPedigreeFunction', 'Insulin')
    columns -= IGNORED_COLUMNS
    print(f"Removing outliers on the columns: {columns}")
    for column in columns:
        df = pd.read_csv(NEW_FILE)
        q1 = df[column].quantile(0.25)
        q3 = df[column].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        df_cleaned = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
        df_cleaned.to_csv(NEW_FILE, index=False)

    # Check integrity of diabetes_dataset.csv
    print(f"Checking integrity of {NEW_FILE}")
    check_integrity(NEW_FILE)
    print(f"Number of errors: {errors}")
    print(f"Number of rows: {count_rows(NEW_FILE)}")
