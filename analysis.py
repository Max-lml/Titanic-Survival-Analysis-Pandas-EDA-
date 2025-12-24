import pandas as pd
import matplotlib.pyplot as plt

# Titanic Survival Analysis
# Goal: Identify factors associated with passenger survival (EDA + aggregation + visualization)

URL = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"


def main() -> None:
    # 1) Load data
    df = pd.read_csv(URL)

    # 2) Quick EDA
    print("Shape:", df.shape)
    print("\nColumns:", list(df.columns))
    print("\nMissing values:\n", df.isna().sum().sort_values(ascending=False))

    # 3) Key questions
    # Q1: Does sex affect survival?
    survival_by_sex = df.groupby("Sex")["Survived"].mean().sort_values(ascending=False)
    print("\nSurvival rate by sex:\n", survival_by_sex)

    # Q2: Does ticket class affect survival?
    survival_by_class = df.groupby("Pclass")["Survived"].mean().sort_index()
    print("\nSurvival rate by class:\n", survival_by_class)

    # Q3: Is age a strong factor?
    mean_age_by_outcome = df.groupby("Survived")["Age"].mean()
    print("\nMean age by outcome (0=No, 1=Yes):\n", mean_age_by_outcome)

    # Q4: Is fare related to survival? (simple comparison)
    fare_by_outcome = df.groupby("Survived")["Fare"].median()
    print("\nMedian fare by outcome (0=No, 1=Yes):\n", fare_by_outcome)

    # 4) Visualizations (2–3 plots, only meaningful ones)
    plt.figure()
    survival_by_sex.plot(kind="bar", title="Survival rate by sex")
    plt.ylabel("Survival rate")
    plt.ylim(0, 1)

    plt.figure()
    survival_by_class.plot(kind="bar", title="Survival rate by ticket class")
    plt.ylabel("Survival rate")
    plt.ylim(0, 1)

    plt.figure()
    # Age distribution for survivors vs non-survivors (drop NaN)
    df[df["Survived"] == 0]["Age"].dropna().plot(kind="hist", bins=20, alpha=0.6, label="Did not survive")
    df[df["Survived"] == 1]["Age"].dropna().plot(kind="hist", bins=20, alpha=0.6, label="Survived")
    plt.title("Age distribution by survival outcome")
    plt.xlabel("Age")
    plt.ylabel("Count")
    plt.legend()

    plt.tight_layout()
    plt.show()

    # 5) Conclusions (rus)
    print("\nВыводы:")
    print("- Женщины выживали заметно чаще мужчин.")
    print("- Чем выше класс, тем выше доля выживших.")
    print("- Средний возраст выживших и погибших близок, поэтому возраст не главный фактор по среднему.")
    print("- Цена билета в среднем выше у выживших (но лучше проверять медианой, если есть выбросы).")


if __name__ == "__main__":
    main()
