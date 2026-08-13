"""
Netflix Data Analysis
======================
Author: Dev Bhardwaj
Description:
    End-to-end analysis of the Netflix Movies and TV Shows dataset.
    Covers data loading, cleaning, exploratory data analysis (EDA),
    and visualization of content trends, genres, countries, ratings,
    durations, and directors.

Run:
    python src/analysis.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 110

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "netflix_titles.csv")
VIZ_DIR = os.path.join(BASE_DIR, "visualizations")
os.makedirs(VIZ_DIR, exist_ok=True)

NETFLIX_RED = "#E50914"
NETFLIX_DARK = "#221F1F"
PALETTE = ["#E50914", "#221F1F", "#B81D24", "#F5F5F1", "#564D4D"]


def savefig(name):
    """Save the current matplotlib figure into the visualizations folder."""
    path = os.path.join(VIZ_DIR, name)
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


# ---------------------------------------------------------------------------
# 1. DATA LOADING
# ---------------------------------------------------------------------------
def load_data():
    df = pd.read_csv(DATA_PATH)

    print("=" * 60)
    print("STEP 1: DATA LOADING")
    print("=" * 60)
    print("\nFirst 5 rows:")
    print(df.head())
    print(f"\nDataset shape: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"\nColumn names: {list(df.columns)}")
    print("\nData types:")
    print(df.dtypes)
    print("\nDescriptive statistics (numeric columns):")
    print(df.describe())

    return df


# ---------------------------------------------------------------------------
# 2. DATA CLEANING
# ---------------------------------------------------------------------------
def clean_data(df):
    print("\n" + "=" * 60)
    print("STEP 2: DATA CLEANING")
    print("=" * 60)

    shape_before = df.shape
    print(f"\nShape before cleaning: {shape_before}")

    # --- Missing values ---
    missing_count = df.isnull().sum()
    missing_pct = (missing_count / len(df) * 100).round(2)
    missing_report = pd.DataFrame(
        {"missing_count": missing_count, "missing_pct": missing_pct}
    )
    missing_report = missing_report[missing_report["missing_count"] > 0].sort_values(
        "missing_count", ascending=False
    )
    print("\nMissing values before cleaning:")
    print(missing_report)

    df = df.copy()

    # --- Remove exact duplicate rows ---
    dup_count = df.duplicated().sum()
    print(f"\nDuplicate rows found: {dup_count}")
    df = df.drop_duplicates()

    # --- Duplicate titles (same title + type) ---
    dup_titles = df.duplicated(subset=["title", "type"]).sum()
    print(f"Duplicate (title, type) combinations found: {dup_titles}")
    df = df.drop_duplicates(subset=["title", "type"])

    # --- Clean whitespace / inconsistent text on key text columns ---
    text_cols = ["type", "title", "director", "cast", "country", "rating", "listed_in"]
    for col in text_cols:
        df[col] = df[col].astype(str).str.strip()
        df[col] = df[col].replace({"nan": np.nan, "": np.nan})

    # --- Handle missing categorical values ---
    df["director"] = df["director"].fillna("Unknown")
    df["cast"] = df["cast"].fillna("Unknown")
    df["country"] = df["country"].fillna("Unknown")
    df["rating"] = df["rating"].fillna(df["rating"].mode()[0])

    # --- date_added: convert to datetime, handle missing ---
    df["date_added"] = df["date_added"].astype(str).str.strip()
    df["date_added"] = pd.to_datetime(df["date_added"], errors="coerce")
    df["year_added"] = df["date_added"].dt.year
    df["month_added"] = df["date_added"].dt.month_name()

    # --- release_year: check for invalid years ---
    current_year = 2026
    invalid_years = df[(df["release_year"] < 1900) | (df["release_year"] > current_year)]
    print(f"\nInvalid release_year values found: {len(invalid_years)}")

    # --- duration: split into numeric value + unit, handle missing ---
    df["duration"] = df["duration"].astype(str).str.strip()
    df.loc[df["duration"] == "nan", "duration"] = np.nan
    missing_duration = df["duration"].isnull().sum()
    print(f"Missing duration values: {missing_duration}")

    def parse_duration(row):
        dur = row["duration"]
        if pd.isna(dur):
            return np.nan
        digits = "".join(ch for ch in dur if ch.isdigit())
        return float(digits) if digits else np.nan

    df["duration_value"] = df.apply(parse_duration, axis=1)

    # Movies -> minutes, TV Shows -> number of seasons
    df["duration_minutes"] = np.where(df["type"] == "Movie", df["duration_value"], np.nan)
    df["duration_seasons"] = np.where(df["type"] == "TV Show", df["duration_value"], np.nan)

    # Fill missing movie duration with the median movie duration
    movie_median = df.loc[df["type"] == "Movie", "duration_minutes"].median()
    df.loc[df["type"] == "Movie", "duration_minutes"] = df.loc[
        df["type"] == "Movie", "duration_minutes"
    ].fillna(movie_median)

    # --- Standardize categorical text (title case for country/type) ---
    df["type"] = df["type"].str.strip()
    df["primary_country"] = df["country"].apply(
        lambda x: x.split(",")[0].strip() if isinstance(x, str) else x
    )

    shape_after = df.shape
    print(f"\nShape after cleaning: {shape_after}")
    print(f"Rows removed during cleaning: {shape_before[0] - shape_after[0]}")

    print("\nMissing values after cleaning (key columns):")
    print(df[["director", "cast", "country", "rating", "duration"]].isnull().sum())

    return df, shape_before, shape_after


# ---------------------------------------------------------------------------
# 3. EXPLORATORY DATA ANALYSIS
# ---------------------------------------------------------------------------
def explode_genres(df):
    """Return a long-format dataframe with one row per (title, genre)."""
    temp = df[["show_id", "type", "listed_in"]].copy()
    temp["genre"] = temp["listed_in"].str.split(",")
    temp = temp.explode("genre")
    temp["genre"] = temp["genre"].str.strip()
    return temp


def explode_countries(df):
    temp = df[["show_id", "type", "country"]].copy()
    temp = temp[temp["country"] != "Unknown"]
    temp["country_split"] = temp["country"].str.split(",")
    temp = temp.explode("country_split")
    temp["country_split"] = temp["country_split"].str.strip()
    return temp


def explode_cast(df):
    temp = df[["show_id", "cast"]].copy()
    temp = temp[temp["cast"] != "Unknown"]
    temp["actor"] = temp["cast"].str.split(",")
    temp = temp.explode("actor")
    temp["actor"] = temp["actor"].str.strip()
    return temp


def run_eda(df):
    print("\n" + "=" * 60)
    print("STEP 3: EXPLORATORY DATA ANALYSIS")
    print("=" * 60)

    results = {}

    # --- Content type ---
    type_counts = df["type"].value_counts()
    type_pct = (type_counts / len(df) * 100).round(2)
    print("\nContent type counts:")
    print(type_counts)
    print("\nContent type percentages:")
    print(type_pct)
    results["type_counts"] = type_counts
    results["type_pct"] = type_pct

    # --- Release trends ---
    year_counts = df["release_year"].value_counts().sort_index()
    results["year_counts"] = year_counts

    year_type_counts = df.groupby(["release_year", "type"]).size().unstack(fill_value=0)
    results["year_type_counts"] = year_type_counts

    added_counts = df["year_added"].value_counts().sort_index()
    results["added_counts"] = added_counts

    # --- Genre analysis ---
    genre_long = explode_genres(df)
    genre_counts = genre_long["genre"].value_counts()
    top10_genres = genre_counts.head(10)
    print("\nTop 10 genres overall:")
    print(top10_genres)
    results["genre_counts"] = genre_counts
    results["top10_genres"] = top10_genres

    genre_by_type = genre_long.groupby(["type", "genre"]).size().unstack(level=0, fill_value=0)
    results["genre_by_type"] = genre_by_type

    # --- Country analysis ---
    country_long = explode_countries(df)
    country_counts = country_long["country_split"].value_counts()
    top10_countries = country_counts.head(10)
    print("\nTop 10 countries:")
    print(top10_countries)
    results["country_counts"] = country_counts
    results["top10_countries"] = top10_countries

    country_type = country_long.groupby(["country_split", "type"]).size().unstack(fill_value=0)
    results["country_type"] = country_type

    # --- Ratings ---
    rating_counts = df["rating"].value_counts()
    print("\nContent rating distribution:")
    print(rating_counts)
    results["rating_counts"] = rating_counts

    rating_by_type = df.groupby(["rating", "type"]).size().unstack(fill_value=0)
    results["rating_by_type"] = rating_by_type

    # --- Movie duration ---
    movie_durations = df.loc[df["type"] == "Movie", "duration_minutes"].dropna()
    print(f"\nMovie duration -> mean: {movie_durations.mean():.1f} min, "
          f"min: {movie_durations.min():.0f} min, max: {movie_durations.max():.0f} min")
    results["movie_durations"] = movie_durations

    # --- TV show seasons ---
    tv_seasons = df.loc[df["type"] == "TV Show", "duration_seasons"].dropna()
    print(f"\nTV Show seasons -> mean: {tv_seasons.mean():.2f}, "
          f"most common: {tv_seasons.mode()[0]:.0f}")
    results["tv_seasons"] = tv_seasons

    # --- Directors ---
    directors = df[df["director"] != "Unknown"]["director"]
    director_split = directors.str.split(",").explode().str.strip()
    top10_directors = director_split.value_counts().head(10)
    print("\nTop 10 directors:")
    print(top10_directors)
    results["top10_directors"] = top10_directors

    # --- Cast ---
    cast_long = explode_cast(df)
    top10_actors = cast_long["actor"].value_counts().head(10)
    print("\nTop 10 most frequent cast members:")
    print(top10_actors)
    results["top10_actors"] = top10_actors

    return results


# ---------------------------------------------------------------------------
# 4. VISUALIZATIONS  (10+ charts saved to /visualizations)
# ---------------------------------------------------------------------------
def make_visualizations(df, res):
    print("\n" + "=" * 60)
    print("STEP 4: VISUALIZATIONS")
    print("=" * 60)

    # 1. Movies vs TV Shows - bar chart
    plt.figure(figsize=(6, 5))
    sns.barplot(x=res["type_counts"].index, y=res["type_counts"].values, palette=PALETTE)
    plt.title("Netflix Content: Movies vs TV Shows", fontsize=13, fontweight="bold")
    plt.xlabel("Content Type")
    plt.ylabel("Number of Titles")
    for i, v in enumerate(res["type_counts"].values):
        plt.text(i, v + 30, str(v), ha="center", fontweight="bold")
    savefig("01_movies_vs_tvshows_bar.png")

    # 2. Movies vs TV Shows - pie/percentage chart
    plt.figure(figsize=(6, 6))
    plt.pie(
        res["type_pct"].values,
        labels=res["type_pct"].index,
        autopct="%1.1f%%",
        colors=[NETFLIX_RED, NETFLIX_DARK],
        startangle=90,
        textprops={"fontsize": 11},
    )
    plt.title("Content Type Distribution (%)", fontsize=13, fontweight="bold")
    savefig("02_movies_vs_tvshows_percentage.png")

    # 3. Titles released by year
    plt.figure(figsize=(11, 5))
    recent = res["year_counts"][res["year_counts"].index >= 1990]
    plt.plot(recent.index, recent.values, color=NETFLIX_RED, linewidth=2)
    plt.fill_between(recent.index, recent.values, color=NETFLIX_RED, alpha=0.15)
    plt.title("Netflix Titles by Release Year", fontsize=13, fontweight="bold")
    plt.xlabel("Release Year")
    plt.ylabel("Number of Titles")
    savefig("03_titles_released_by_year.png")

    # 4. Movie vs TV Show release trend over years
    plt.figure(figsize=(11, 5))
    yt = res["year_type_counts"]
    yt = yt[yt.index >= 1990]
    plt.plot(yt.index, yt["Movie"], label="Movies", color=NETFLIX_RED, linewidth=2)
    plt.plot(yt.index, yt["TV Show"], label="TV Shows", color=NETFLIX_DARK, linewidth=2)
    plt.title("Movie vs TV Show Releases by Year", fontsize=13, fontweight="bold")
    plt.xlabel("Release Year")
    plt.ylabel("Number of Titles")
    plt.legend()
    savefig("04_movie_vs_tv_release_trend.png")

    # 5. Top 10 genres
    plt.figure(figsize=(9, 6))
    sns.barplot(x=res["top10_genres"].values, y=res["top10_genres"].index, palette="Reds_r")
    plt.title("Top 10 Genres on Netflix", fontsize=13, fontweight="bold")
    plt.xlabel("Number of Titles")
    plt.ylabel("Genre")
    savefig("05_top10_genres.png")

    # 6. Top 10 countries
    plt.figure(figsize=(9, 6))
    sns.barplot(x=res["top10_countries"].values, y=res["top10_countries"].index, palette="Reds_r")
    plt.title("Top 10 Content-Producing Countries", fontsize=13, fontweight="bold")
    plt.xlabel("Number of Titles")
    plt.ylabel("Country")
    savefig("06_top10_countries.png")

    # 7. Content rating distribution
    plt.figure(figsize=(10, 5))
    order = res["rating_counts"].index
    sns.barplot(x=order, y=res["rating_counts"].values, palette="Reds_r")
    plt.title("Content Rating Distribution", fontsize=13, fontweight="bold")
    plt.xlabel("Rating")
    plt.ylabel("Number of Titles")
    plt.xticks(rotation=45)
    savefig("07_rating_distribution.png")

    # 8. Movie duration distribution
    plt.figure(figsize=(9, 5))
    sns.histplot(res["movie_durations"], bins=30, color=NETFLIX_RED, kde=True)
    plt.title("Movie Duration Distribution", fontsize=13, fontweight="bold")
    plt.xlabel("Duration (minutes)")
    plt.ylabel("Count")
    savefig("08_movie_duration_distribution.png")

    # 9. TV Show season distribution
    plt.figure(figsize=(9, 5))
    season_counts = res["tv_seasons"].value_counts().sort_index()
    sns.barplot(x=season_counts.index.astype(int), y=season_counts.values, palette="Reds_r")
    plt.title("TV Show Season Count Distribution", fontsize=13, fontweight="bold")
    plt.xlabel("Number of Seasons")
    plt.ylabel("Number of TV Shows")
    savefig("09_tvshow_season_distribution.png")

    # 10. Top directors
    plt.figure(figsize=(9, 6))
    sns.barplot(x=res["top10_directors"].values, y=res["top10_directors"].index, palette="Reds_r")
    plt.title("Top 10 Directors by Number of Titles", fontsize=13, fontweight="bold")
    plt.xlabel("Number of Titles")
    plt.ylabel("Director")
    savefig("10_top10_directors.png")

    # 11. Rating vs content type (stacked)
    plt.figure(figsize=(10, 6))
    res["rating_by_type"].plot(kind="bar", stacked=True, color=[NETFLIX_RED, NETFLIX_DARK], ax=plt.gca())
    plt.title("Content Rating by Type (Movie vs TV Show)", fontsize=13, fontweight="bold")
    plt.xlabel("Rating")
    plt.ylabel("Number of Titles")
    plt.xticks(rotation=45)
    plt.legend(title="Type")
    savefig("11_rating_by_content_type.png")

    # 12. Content added to Netflix per year (growth over time)
    plt.figure(figsize=(10, 5))
    added = res["added_counts"].dropna()
    added = added[added.index >= 2008]
    plt.bar(added.index.astype(int), added.values, color=NETFLIX_RED)
    plt.title("Growth of Netflix Content Library (Titles Added per Year)", fontsize=13, fontweight="bold")
    plt.xlabel("Year Added to Netflix")
    plt.ylabel("Number of Titles Added")
    savefig("12_content_added_growth.png")

    # 13. Top 10 countries by content type (grouped)
    plt.figure(figsize=(10, 6))
    top_c = res["top10_countries"].index
    ct = res["country_type"].loc[top_c]
    ct.plot(kind="barh", stacked=True, color=[NETFLIX_RED, NETFLIX_DARK], ax=plt.gca())
    plt.title("Top 10 Countries: Movie vs TV Show Split", fontsize=13, fontweight="bold")
    plt.xlabel("Number of Titles")
    plt.ylabel("Country")
    plt.legend(title="Type")
    savefig("13_country_content_type_split.png")

    # 14. Top 10 genres: movies vs TV shows
    plt.figure(figsize=(10, 6))
    top_genres = res["top10_genres"].index
    gbt = res["genre_by_type"].reindex(top_genres).fillna(0)
    gbt.plot(kind="barh", stacked=True, color=[NETFLIX_RED, NETFLIX_DARK], ax=plt.gca())
    plt.title("Top 10 Genres: Movies vs TV Shows", fontsize=13, fontweight="bold")
    plt.xlabel("Number of Titles")
    plt.ylabel("Genre")
    plt.legend(title="Type")
    savefig("14_genre_by_content_type.png")

    # 15. Correlation heatmap of numeric variables
    plt.figure(figsize=(6, 5))
    numeric_df = df[["release_year", "duration_minutes", "duration_seasons", "year_added"]]
    corr = numeric_df.corr()
    sns.heatmap(corr, annot=True, cmap="Reds", fmt=".2f", linewidths=0.5)
    plt.title("Correlation Heatmap: Numeric Variables", fontsize=13, fontweight="bold")
    savefig("15_correlation_heatmap.png")

    print(f"\nTotal visualizations saved: 15 (in /{os.path.relpath(VIZ_DIR, BASE_DIR)})")


# ---------------------------------------------------------------------------
# 5. STATISTICAL SUMMARY
# ---------------------------------------------------------------------------
def statistical_summary(df, res):
    print("\n" + "=" * 60)
    print("STEP 5: STATISTICAL SUMMARY")
    print("=" * 60)

    total_titles = len(df)
    total_movies = int(res["type_counts"].get("Movie", 0))
    total_tv = int(res["type_counts"].get("TV Show", 0))
    movie_pct = round(total_movies / total_titles * 100, 2)
    tv_pct = round(total_tv / total_titles * 100, 2)
    unique_genres = df["listed_in"].str.split(",").explode().str.strip().nunique()
    unique_countries = res["country_counts"].shape[0]
    avg_movie_duration = round(res["movie_durations"].mean(), 1)
    avg_tv_seasons = round(res["tv_seasons"].mean(), 2)
    earliest_year = int(df["release_year"].min())
    latest_year = int(df["release_year"].max())

    summary = {
        "Total titles": total_titles,
        "Total movies": total_movies,
        "Total TV shows": total_tv,
        "Movie percentage": f"{movie_pct}%",
        "TV Show percentage": f"{tv_pct}%",
        "Number of unique genres": unique_genres,
        "Number of countries": unique_countries,
        "Average movie duration (min)": avg_movie_duration,
        "Average TV show seasons": avg_tv_seasons,
        "Earliest release year": earliest_year,
        "Latest release year": latest_year,
    }

    print()
    for k, v in summary.items():
        print(f"{k}: {v}")

    return summary


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    df = load_data()
    df, shape_before, shape_after = clean_data(df)
    res = run_eda(df)
    make_visualizations(df, res)
    summary = statistical_summary(df, res)

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    return df, res, summary


if __name__ == "__main__":
    main()
