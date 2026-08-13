# Netflix Data Analysis | Python, Pandas, Matplotlib, Seaborn

An end-to-end exploratory data analysis (EDA) project on the Netflix Movies and TV Shows catalog, covering data cleaning, genre and country analysis, release trends, and content-preference insights — with 15 visualizations built in Matplotlib and Seaborn.

**Project timeline:** January 2026 – February 2026

---

## 1. Project Overview

Netflix's content library spans thousands of movies and TV shows across genres, countries, and decades. This project analyzes the full public **Netflix Movies and TV Shows dataset (8,807 titles)** to answer questions such as:

- Does Netflix's catalog lean toward Movies or TV Shows?
- Which genres and countries dominate the platform?
- How has content output grown over time?
- What do content ratings, movie durations, and season counts reveal about the platform's audience strategy?

## 2. Objective

- Clean and structure a real-world, messy dataset using Python, Pandas, and NumPy.
- Perform genre, country, rating, duration, and director/cast analysis.
- Build 10+ publication-quality visualizations.
- Translate statistical findings into plain-language, data-driven insights.

## 3. Dataset Information

| Detail | Value |
|---|---|
| Source | Netflix Movies and TV Shows dataset (public dataset, originally published on Kaggle by Shivam Bansal) |
| Records analyzed | **8,807 titles** |
| Columns | `show_id`, `type`, `title`, `director`, `cast`, `country`, `date_added`, `release_year`, `rating`, `duration`, `listed_in`, `description` |
| File | `data/netflix_titles.csv` |

> The dataset reflects Netflix's catalog as captured at the time of collection and does not represent Netflix's real-time current library.

## 4. Technologies Used

- **Python 3**
- **Pandas** — data loading, cleaning, transformation, group-by analysis
- **NumPy** — numeric operations and missing-value handling
- **Matplotlib** — chart rendering
- **Seaborn** — statistical visualizations and styling
- **Jupyter Notebook** — analysis walkthrough

## 5. Project Workflow

1. **Data Loading** — import the CSV, inspect shape, dtypes, and summary statistics.
2. **Data Cleaning** — handle missing values, remove duplicates, parse dates, standardize text, validate release years, and parse duration into numeric fields.
3. **Exploratory Data Analysis** — content type, release trends, genre analysis, country analysis, ratings, movie duration, TV season counts, directors, and cast.
4. **Visualization** — 15 Matplotlib/Seaborn charts saved to `/visualizations`.
5. **Insight Generation** — plain-language business insights derived directly from the computed statistics.

## 6. Data Cleaning Process

| Step | Result |
|---|---|
| Missing values detected | `director` 29.91%, `country` 9.44%, `cast` 9.37%, `date_added` 0.11%, `rating` 0.05%, `duration` 0.03% |
| Duplicate rows | 0 exact duplicates found |
| Duplicate (title, type) pairs | 0 found |
| Missing `director` / `cast` / `country` | Filled with `"Unknown"` to preserve every record |
| Missing `rating` | Filled with the most frequent rating (mode) |
| `date_added` | Converted to `datetime`; `year_added` and `month_added` derived |
| Invalid `release_year` values (before 1900 or after 2026) | 0 found |
| `duration` | Split into `duration_minutes` (Movies) and `duration_seasons` (TV Shows); missing movie durations filled with the median |
| Shape before cleaning | (8807, 12) |
| Shape after cleaning | (8807, 18) — 0 rows dropped, 6 engineered columns added |

No rows were dropped during cleaning — all missing data was handled through informed imputation rather than deletion, preserving the full 8,807-record dataset.

## 7. Exploratory Analysis

- **Content Type** — Movies vs TV Shows, counts and percentages
- **Release Trends** — titles per year, Movie vs TV Show release trend, content-library growth by year added
- **Genre Analysis** — top genres overall, and split by Movie vs TV Show
- **Country Analysis** — top content-producing countries, and country vs content-type split
- **Ratings** — rating distribution, and rating vs content type
- **Movie Duration** — distribution, mean/min/max, most common range
- **TV Show Seasons** — distribution, mean, most common season count
- **Directors** — top 10 most prolific directors
- **Cast** — top 10 most frequently appearing actors
- **Correlation Heatmap** — relationships between numeric variables

## 8. List of Visualizations (15 total)

All saved to `/visualizations`:

1. `01_movies_vs_tvshows_bar.png` — Movies vs TV Shows bar chart
2. `02_movies_vs_tvshows_percentage.png` — Movies vs TV Shows percentage (pie) chart
3. `03_titles_released_by_year.png` — Netflix titles released by year
4. `04_movie_vs_tv_release_trend.png` — Movie vs TV Show release trend over time
5. `05_top10_genres.png` — Top 10 genres
6. `06_top10_countries.png` — Top 10 content-producing countries
7. `07_rating_distribution.png` — Content rating distribution
8. `08_movie_duration_distribution.png` — Movie duration distribution
9. `09_tvshow_season_distribution.png` — TV Show season count distribution
10. `10_top10_directors.png` — Top 10 directors by number of titles
11. `11_rating_by_content_type.png` — Rating vs content type (stacked)
12. `12_content_added_growth.png` — Growth of Netflix content library by year added
13. `13_country_content_type_split.png` — Top 10 countries: Movie vs TV Show split
14. `14_genre_by_content_type.png` — Top 10 genres: Movie vs TV Show split
15. `15_correlation_heatmap.png` — Correlation heatmap of numeric variables

## 9. Final Calculated Statistics

| Metric | Value |
|---|---|
| Total titles | **8,807** |
| Total movies | **6,131** |
| Total TV shows | **2,676** |
| Movie percentage | **69.62%** |
| TV Show percentage | **30.38%** |
| Number of unique genres | **42** |
| Number of countries represented | **123** |
| Average movie duration | **99.6 minutes** |
| Average TV show seasons | **1.76** |
| Most common TV show season count | **1 season** |
| Earliest release year | **1925** |
| Latest release year | **2021** |
| Top genre | **International Movies** |
| Top content-producing country | **United States** (3,690 titles) |
| Most common content rating | **TV-MA** (3,211 titles) |
| Most prolific director | **Rajiv Chilaka** (22 titles) |

## 10. Key Findings

1. **Movies dominate the library** — ~70% of titles (6,131) are Movies vs ~30% (2,676) TV Shows.
2. **International Movies and Dramas are the leading genres**, reflecting a global-content acquisition strategy.
3. **The United States leads content production** (3,690 titles), followed by India (1,046) and the United Kingdom (806).
4. **Content additions grew sharply from 2015–2019** before declining in the most recent years covered by the dataset.
5. **TV-MA is the most common content rating**, indicating the catalog skews toward mature/adult audiences over family programming.
6. **The average movie runs ~100 minutes**, consistent with standard theatrical run-times, ranging from 3 to 312 minutes.
7. **Most TV shows have only 1 season** (average 1.76 seasons), suggesting a large share of limited series or shows not renewed.
8. **Indian talent dominates the most-frequent-cast list** (Anupam Kher, Shah Rukh Khan, Naseeruddin Shah, and others), reflecting the scale of Indian content in the catalog.
9. **Regional-content directors** (Rajiv Chilaka, Jan Suter, Raúl Campos) top the director list — more than any single Hollywood director — showing investment in high-output regional partners.
10. **TV Show output has grown faster than Movie output in recent years**, gradually narrowing the historical content-type gap even though Movies remain dominant overall.

## 11. Project Structure

```
Netflix-Data-Analysis/
│
├── data/
│   └── netflix_titles.csv
│
├── notebooks/
│   └── Netflix_Data_Analysis.ipynb
│
├── src/
│   └── analysis.py
│
├── visualizations/
│   └── (15 PNG charts)
│
├── README.md
├── requirements.txt
└── .gitignore
```

## 12. How to Run the Project

```bash
# 1. Clone the repository
git clone <repository-url>
cd Netflix-Data-Analysis

# 2. Install dependencies
pip install -r requirements.txt

# 3a. Run the full analysis as a script (saves all charts to /visualizations)
python src/analysis.py

# 3b. OR explore interactively via Jupyter Notebook
jupyter notebook notebooks/Netflix_Data_Analysis.ipynb
```

## 13. Future Improvements

- Integrate IMDb/Rotten Tomatoes ratings to correlate content ratings with audience reception.
- Build an interactive dashboard (Plotly Dash / Streamlit / Power BI) on top of the cleaned dataset.
- Apply NLP on the `description` field to cluster content by theme/sentiment.
- Track a live Netflix catalog feed to extend the release-trend analysis beyond 2021.
- Add genre-level year-over-year trend analysis to spot rising/declining content categories.

## 14. Resume-Ready Summary

- Analyzed **8,807 Netflix titles** to identify genre distribution, release trends, and content patterns, uncovering key insights into the platform's content library.
- Cleaned and transformed the dataset using **Python, Pandas, and NumPy**, handling missing values across 6 columns and engineering 6 new analysis-ready fields without dropping any records.
- Created **15 visualizations** using **Matplotlib** and **Seaborn** to highlight content trends and support data-driven insights.

---

*All statistics and findings in this README were computed directly from the dataset in `data/netflix_titles.csv` — no numbers were fabricated or estimated.*
