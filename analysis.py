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

    # 5) Conclusions (short, human-readable)
    print("\nConclusions:")
    print(f"- Sex is a strong factor: female survival rate ≈ {survival_by_sex.get('female', float('nan')):.2f}, "
          f"male ≈ {survival_by_sex.get('male', float('nan')):.2f}.")
    print(f"- Ticket class matters: 1st class ≈ {survival_by_class.loc[1]:.2f}, "
          f"2nd ≈ {survival_by_class.loc[2]:.2f}, 3rd ≈ {survival_by_class.loc[3]:.2f}.")
    print("- Age effect is weaker in terms of mean age, but distributions can differ (see histogram).")
    print("- Higher ticket fares tend to be associated with better outcomes (median fare is higher for survivors).")


if __name__ == "__main__":
    main()
