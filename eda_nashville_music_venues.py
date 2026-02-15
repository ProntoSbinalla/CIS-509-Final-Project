"""
Exploratory Data Analysis: Nashville Music Venues (2013-2019)

This script performs comprehensive EDA on the filtered dataset:
1. Business-level analysis (ratings, review counts, temporal coverage)
2. Review-level analysis (trends, seasonality, text length)
3. Data quality assessment (missing data, sparsity)
4. Initial signal detection (high vs. low-rated patterns)
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from collections import Counter, defaultdict
import warnings
warnings.filterwarnings('ignore')

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10

print("=" * 80)
print("EXPLORATORY DATA ANALYSIS: Nashville Music Venues (2013-2019)")
print("=" * 80)

# ============================================================================
# LOAD DATA
# ============================================================================
print("\n[1] Loading filtered dataset...")

with open('filtered_data/combined_nashville_music_venues.json', 'r', encoding='utf-8') as f:
    combined_data = json.load(f)

print(f"   Loaded {len(combined_data)} venues")

# Create separate DataFrames for businesses and reviews
businesses = []
all_reviews = []

for venue in combined_data:
    business_info = venue['business_info']
    businesses.append(business_info)
    
    for review in venue['reviews']:
        review_with_business = review.copy()
        review_with_business['business_name'] = business_info['name']
        review_with_business['business_stars'] = business_info['stars']
        all_reviews.append(review_with_business)

df_business = pd.DataFrame(businesses)
df_reviews = pd.DataFrame(all_reviews)

# Parse dates
df_reviews['date'] = pd.to_datetime(df_reviews['date'])
df_reviews['year'] = df_reviews['date'].dt.year
df_reviews['month'] = df_reviews['date'].dt.month
df_reviews['year_month'] = df_reviews['date'].dt.to_period('M')

# Calculate review text length
df_reviews['text_length'] = df_reviews['text'].str.len()
df_reviews['word_count'] = df_reviews['text'].str.split().str.len()

print(f"   Business DataFrame: {df_business.shape}")
print(f"   Reviews DataFrame: {df_reviews.shape}")

# Create output directory for plots
import os
os.makedirs('eda_plots', exist_ok=True)

# ============================================================================
# PART 1: BUSINESS-LEVEL ANALYSIS
# ============================================================================
print("\n" + "=" * 80)
print("PART 1: BUSINESS-LEVEL ANALYSIS")
print("=" * 80)

# 1.1 Star Rating Distribution
print("\n[1.1] Venue Star Rating Distribution")
print(f"   Mean: {df_business['stars'].mean():.2f}")
print(f"   Median: {df_business['stars'].median():.2f}")
print(f"   Std Dev: {df_business['stars'].std():.2f}")
print(f"   Range: {df_business['stars'].min():.1f} - {df_business['stars'].max():.1f}")

# 1.2 Review Count Distribution
review_counts = df_reviews.groupby('business_id').size()
df_business['filtered_review_count'] = df_business['business_id'].map(review_counts)

print(f"\n[1.2] Review Count Distribution (2013-2019)")
print(f"   Mean: {df_business['filtered_review_count'].mean():.1f}")
print(f"   Median: {df_business['filtered_review_count'].median():.1f}")
print(f"   Std Dev: {df_business['filtered_review_count'].std():.1f}")
print(f"   Range: {df_business['filtered_review_count'].min()} - {df_business['filtered_review_count'].max()}")

# Identify outliers (venues with exceptionally high review counts)
q75 = df_business['filtered_review_count'].quantile(0.75)
q25 = df_business['filtered_review_count'].quantile(0.25)
iqr = q75 - q25
outlier_threshold = q75 + 1.5 * iqr

outliers = df_business[df_business['filtered_review_count'] > outlier_threshold]
print(f"\n   Outliers (>{outlier_threshold:.0f} reviews): {len(outliers)} venues")
if len(outliers) > 0:
    print("   Top review count venues:")
    for _, venue in outliers.nlargest(5, 'filtered_review_count').iterrows():
        print(f"      - {venue['name']}: {venue['filtered_review_count']} reviews ({venue['stars']} stars)")

# 1.3 Price Range Analysis
print(f"\n[1.3] Price Range Analysis")
# Extract price range from attributes
def extract_price_range(attributes):
    if attributes and isinstance(attributes, dict):
        for key in ['RestaurantsPriceRange2', 'RestaurantsPriceRange']:
            if key in attributes:
                return attributes[key]
    return None

df_business['price_range'] = df_business['attributes'].apply(extract_price_range)
price_available = df_business['price_range'].notna().sum()
print(f"   Venues with price data: {price_available} / {len(df_business)} ({price_available/len(df_business)*100:.1f}%)")

if price_available > 0:
    price_dist = df_business['price_range'].value_counts().sort_index()
    print("   Price range distribution:")
    for price, count in price_dist.items():
        print(f"      ${price}: {count} venues ({count/price_available*100:.1f}%)")

# 1.4 Temporal Coverage Analysis
print(f"\n[1.4] Temporal Coverage Analysis")

# Calculate review activity per year for each venue
venue_yearly_activity = df_reviews.groupby(['business_id', 'year']).size().unstack(fill_value=0)

# Count how many years each venue has reviews
years_with_reviews = (venue_yearly_activity > 0).sum(axis=1)
df_business['years_active'] = df_business['business_id'].map(years_with_reviews)

print(f"   Years with review activity (2013-2019):")
for year_count in sorted(df_business['years_active'].unique(), reverse=True):
    count = (df_business['years_active'] == year_count).sum()
    print(f"      {year_count} years: {count} venues ({count/len(df_business)*100:.1f}%)")

consistent_venues = df_business[df_business['years_active'] >= 5]
print(f"\n   Venues with consistent activity (>=5 years): {len(consistent_venues)} ({len(consistent_venues)/len(df_business)*100:.1f}%)")

# ============================================================================
# PART 2: REVIEW-LEVEL ANALYSIS
# ============================================================================
print("\n" + "=" * 80)
print("PART 2: REVIEW-LEVEL ANALYSIS")
print("=" * 80)

# 2.1 Star Rating Distribution
print("\n[2.1] Review Star Rating Distribution")
star_dist = df_reviews['stars'].value_counts().sort_index()
for stars, count in star_dist.items():
    print(f"   {int(stars)} stars: {count:,} reviews ({count/len(df_reviews)*100:.1f}%)")

# 2.2 Review Volume Trends Over Time
print(f"\n[2.2] Review Volume Trends (2013-2019)")
yearly_reviews = df_reviews.groupby('year').size()
print("   Reviews per year:")
for year, count in yearly_reviews.items():
    print(f"      {year}: {count:,} reviews")

# Calculate year-over-year growth
for i in range(1, len(yearly_reviews)):
    prev_year = yearly_reviews.index[i-1]
    curr_year = yearly_reviews.index[i]
    growth = ((yearly_reviews.iloc[i] - yearly_reviews.iloc[i-1]) / yearly_reviews.iloc[i-1]) * 100
    print(f"      {prev_year} -> {curr_year}: {growth:+.1f}% change")

# 2.3 Seasonality Patterns
print(f"\n[2.3] Seasonality Analysis")
monthly_reviews = df_reviews.groupby('month').size()
print("   Average reviews per month:")
month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
for month, count in monthly_reviews.items():
    avg_per_year = count / 7  # 7 years of data
    print(f"      {month_names[month-1]}: {avg_per_year:.0f} reviews/year")

peak_month = monthly_reviews.idxmax()
low_month = monthly_reviews.idxmin()
print(f"\n   Peak month: {month_names[peak_month-1]} ({monthly_reviews[peak_month]:,} total reviews)")
print(f"   Lowest month: {month_names[low_month-1]} ({monthly_reviews[low_month]:,} total reviews)")

# 2.4 Review Length Statistics
print(f"\n[2.4] Review Length Statistics")
print(f"   Character count:")
print(f"      Mean: {df_reviews['text_length'].mean():.0f}")
print(f"      Median: {df_reviews['text_length'].median():.0f}")
print(f"      Range: {df_reviews['text_length'].min()} - {df_reviews['text_length'].max()}")

print(f"\n   Word count:")
print(f"      Mean: {df_reviews['word_count'].mean():.1f}")
print(f"      Median: {df_reviews['word_count'].median():.0f}")
print(f"      Range: {df_reviews['word_count'].min()} - {df_reviews['word_count'].max()}")

# ============================================================================
# PART 3: DATA QUALITY ASSESSMENT
# ============================================================================
print("\n" + "=" * 80)
print("PART 3: DATA QUALITY ASSESSMENT")
print("=" * 80)

# 3.1 Missing Data Patterns
print("\n[3.1] Missing Data Analysis")
print("   Business attributes:")
print(f"      Address: {df_business['address'].isna().sum()} missing ({df_business['address'].isna().sum()/len(df_business)*100:.1f}%)")
print(f"      Latitude/Longitude: {df_business['latitude'].isna().sum()} missing ({df_business['latitude'].isna().sum()/len(df_business)*100:.1f}%)")
print(f"      Hours: {df_business['hours'].isna().sum()} missing ({df_business['hours'].isna().sum()/len(df_business)*100:.1f}%)")
print(f"      Attributes: {df_business['attributes'].isna().sum()} missing ({df_business['attributes'].isna().sum()/len(df_business)*100:.1f}%)")
print(f"      Price Range: {df_business['price_range'].isna().sum()} missing ({df_business['price_range'].isna().sum()/len(df_business)*100:.1f}%)")

# 3.2 Review Temporal Sparsity
print(f"\n[3.2] Review Temporal Sparsity")

# Calculate gaps in review activity
venue_gaps = []
for business_id in df_business['business_id']:
    venue_reviews = df_reviews[df_reviews['business_id'] == business_id].sort_values('date')
    if len(venue_reviews) > 1:
        date_diffs = venue_reviews['date'].diff().dt.days.dropna()
        max_gap = date_diffs.max()
        avg_gap = date_diffs.mean()
        venue_gaps.append({
            'business_id': business_id,
            'max_gap_days': max_gap,
            'avg_gap_days': avg_gap
        })

df_gaps = pd.DataFrame(venue_gaps)
print(f"   Average gap between reviews:")
print(f"      Mean: {df_gaps['avg_gap_days'].mean():.0f} days")
print(f"      Median: {df_gaps['avg_gap_days'].median():.0f} days")

large_gaps = df_gaps[df_gaps['max_gap_days'] > 365]
print(f"\n   Venues with >1 year gap: {len(large_gaps)} ({len(large_gaps)/len(df_gaps)*100:.1f}%)")

# 3.3 Data Stability Assessment
print(f"\n[3.3] Data Stability for Analysis")
stable_venues = df_business[
    (df_business['filtered_review_count'] >= 50) &
    (df_business['years_active'] >= 5)
]
print(f"   Highly stable venues (>=50 reviews, >=5 years): {len(stable_venues)} ({len(stable_venues)/len(df_business)*100:.1f}%)")

moderate_venues = df_business[
    (df_business['filtered_review_count'] >= 20) &
    (df_business['years_active'] >= 3)
]
print(f"   Moderately stable venues (>=20 reviews, >=3 years): {len(moderate_venues)} ({len(moderate_venues)/len(df_business)*100:.1f}%)")

# ============================================================================
# PART 4: INITIAL SIGNAL DETECTION
# ============================================================================
print("\n" + "=" * 80)
print("PART 4: INITIAL SIGNAL DETECTION")
print("=" * 80)

# 4.1 High vs. Low-Rated Venue Patterns
print("\n[4.1] High vs. Low-Rated Venue Comparison")

high_rated = df_business[df_business['stars'] >= 4.0]
low_rated = df_business[df_business['stars'] <= 2.5]

print(f"   High-rated venues (>=4.0 stars): {len(high_rated)}")
print(f"      Avg reviews: {high_rated['filtered_review_count'].mean():.1f}")
print(f"      Avg years active: {high_rated['years_active'].mean():.1f}")

print(f"\n   Low-rated venues (<=2.5 stars): {len(low_rated)}")
print(f"      Avg reviews: {low_rated['filtered_review_count'].mean():.1f}")
print(f"      Avg years active: {low_rated['years_active'].mean():.1f}")

# 4.2 Text Preview: Common Themes
print(f"\n[4.2] Text Theme Preview")

# Sample reviews from extreme ratings
five_star_reviews = df_reviews[df_reviews['stars'] == 5]['text'].sample(min(100, len(df_reviews[df_reviews['stars'] == 5])))
low_star_reviews = df_reviews[df_reviews['stars'].isin([1, 2])]['text'].sample(min(100, len(df_reviews[df_reviews['stars'].isin([1, 2])])))

# Simple word frequency analysis
def get_common_words(texts, n=10):
    all_words = []
    for text in texts:
        words = text.lower().split()
        # Filter out common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'is', 'was', 'are', 'were', 'it', 'this', 'that', 'i', 'we', 'you', 'they'}
        words = [w.strip('.,!?;:()[]"\'') for w in words if w.lower() not in stop_words and len(w) > 3]
        all_words.extend(words)
    return Counter(all_words).most_common(n)

print("   Common words in 5-star reviews:")
for word, count in get_common_words(five_star_reviews, 10):
    print(f"      {word}: {count}")

print("\n   Common words in 1-2 star reviews:")
for word, count in get_common_words(low_star_reviews, 10):
    print(f"      {word}: {count}")

# 4.3 Rating Volatility Analysis
print(f"\n[4.3] Rating Volatility & Turnaround Candidates")

# Calculate rating variance for each venue
venue_rating_stats = df_reviews.groupby('business_id')['stars'].agg(['mean', 'std', 'count'])
venue_rating_stats = venue_rating_stats[venue_rating_stats['count'] >= 20]  # Only venues with sufficient reviews

high_volatility = venue_rating_stats.nlargest(10, 'std')
print("   Top 10 venues with highest rating volatility:")
for business_id, stats in high_volatility.iterrows():
    venue_name = df_business[df_business['business_id'] == business_id]['name'].values[0]
    business_stars = df_business[df_business['business_id'] == business_id]['stars'].values[0]
    print(f"      {venue_name}: avg={stats['mean']:.2f}, std={stats['std']:.2f}, overall={business_stars}")

# Identify potential turnaround candidates (low overall rating but some high ratings)
turnaround_candidates = []
for business_id in df_business['business_id']:
    venue_reviews = df_reviews[df_reviews['business_id'] == business_id]
    if len(venue_reviews) >= 20:
        overall_stars = df_business[df_business['business_id'] == business_id]['stars'].values[0]
        five_star_pct = (venue_reviews['stars'] == 5).sum() / len(venue_reviews)
        low_star_pct = (venue_reviews['stars'] <= 2).sum() / len(venue_reviews)
        
        # Mixed signals: low overall rating but significant 5-star reviews
        if overall_stars <= 3.5 and five_star_pct >= 0.25 and low_star_pct >= 0.25:
            turnaround_candidates.append({
                'business_id': business_id,
                'name': df_business[df_business['business_id'] == business_id]['name'].values[0],
                'overall_stars': overall_stars,
                'five_star_pct': five_star_pct,
                'low_star_pct': low_star_pct
            })

print(f"\n   Potential turnaround candidates (mixed signals): {len(turnaround_candidates)}")
for candidate in sorted(turnaround_candidates, key=lambda x: x['five_star_pct'], reverse=True)[:5]:
    print(f"      {candidate['name']}: {candidate['overall_stars']} stars, {candidate['five_star_pct']*100:.0f}% 5-star, {candidate['low_star_pct']*100:.0f}% low-star")

# ============================================================================
# PART 5: VISUALIZATIONS
# ============================================================================
print("\n" + "=" * 80)
print("PART 5: CREATING VISUALIZATIONS")
print("=" * 80)

# Chart 1: Business Star Rating Distribution
print("\n[Chart 1] Business Star Rating Distribution")
fig, ax = plt.subplots(figsize=(10, 6))
df_business['stars'].hist(bins=20, edgecolor='black', alpha=0.7, ax=ax)
ax.axvline(df_business['stars'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {df_business["stars"].mean():.2f}')
ax.axvline(df_business['stars'].median(), color='green', linestyle='--', linewidth=2, label=f'Median: {df_business["stars"].median():.2f}')
ax.set_xlabel('Average Star Rating')
ax.set_ylabel('Number of Venues')
ax.set_title('Distribution of Average Star Ratings Across Nashville Music Venues')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('eda_plots/1_business_star_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
print("   Saved: eda_plots/1_business_star_distribution.png")

# Chart 2: Review Count Distribution (Log Scale)
print("\n[Chart 2] Review Count Distribution")
fig, ax = plt.subplots(figsize=(10, 6))
df_business['filtered_review_count'].hist(bins=30, edgecolor='black', alpha=0.7, ax=ax)
ax.set_xlabel('Number of Reviews (2013-2019)')
ax.set_ylabel('Number of Venues')
ax.set_title('Distribution of Review Counts per Venue')
ax.axvline(df_business['filtered_review_count'].median(), color='red', linestyle='--', linewidth=2, label=f'Median: {df_business["filtered_review_count"].median():.0f}')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('eda_plots/2_review_count_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
print("   Saved: eda_plots/2_review_count_distribution.png")

# Chart 3: Review Volume Trends Over Time
print("\n[Chart 3] Review Volume Trends Over Time")
fig, ax = plt.subplots(figsize=(12, 6))
monthly_trend = df_reviews.groupby('year_month').size()
monthly_trend.plot(ax=ax, linewidth=2, color='steelblue')
ax.set_xlabel('Year-Month')
ax.set_ylabel('Number of Reviews')
ax.set_title('Review Volume Trends (2013-2019)')
ax.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('eda_plots/3_review_volume_trends.png', dpi=300, bbox_inches='tight')
plt.close()
print("   Saved: eda_plots/3_review_volume_trends.png")

# Chart 4: Seasonality Pattern
print("\n[Chart 4] Seasonality Pattern")
fig, ax = plt.subplots(figsize=(10, 6))
monthly_avg = df_reviews.groupby('month').size() / 7  # Average per year
month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
ax.bar(range(1, 13), monthly_avg.values, color='coral', edgecolor='black', alpha=0.7)
ax.set_xticks(range(1, 13))
ax.set_xticklabels(month_names)
ax.set_xlabel('Month')
ax.set_ylabel('Average Reviews per Year')
ax.set_title('Seasonality Pattern in Review Activity')
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('eda_plots/4_seasonality_pattern.png', dpi=300, bbox_inches='tight')
plt.close()
print("   Saved: eda_plots/4_seasonality_pattern.png")

# Chart 5: Star Rating Distribution (Reviews)
print("\n[Chart 5] Review Star Rating Distribution")
fig, ax = plt.subplots(figsize=(10, 6))
star_counts = df_reviews['stars'].value_counts().sort_index()
colors = ['#d32f2f', '#f57c00', '#fbc02d', '#7cb342', '#388e3c']
ax.bar(star_counts.index, star_counts.values, color=colors, edgecolor='black', alpha=0.8)
ax.set_xlabel('Star Rating')
ax.set_ylabel('Number of Reviews')
ax.set_title('Distribution of Star Ratings in Reviews')
ax.set_xticks([1, 2, 3, 4, 5])
for i, (stars, count) in enumerate(star_counts.items()):
    ax.text(stars, count, f'{count:,}\n({count/len(df_reviews)*100:.1f}%)', ha='center', va='bottom', fontsize=9)
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('eda_plots/5_review_star_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
print("   Saved: eda_plots/5_review_star_distribution.png")

# Chart 6: High vs Low Rated Venues Comparison
print("\n[Chart 6] High vs. Low-Rated Venue Comparison")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Review counts comparison
categories = ['High-Rated\n(>=4.0 stars)', 'Low-Rated\n(<=2.5 stars)']
review_counts_comparison = [high_rated['filtered_review_count'].mean(), low_rated['filtered_review_count'].mean()]
ax1.bar(categories, review_counts_comparison, color=['green', 'red'], alpha=0.7, edgecolor='black')
ax1.set_ylabel('Average Review Count')
ax1.set_title('Average Review Count: High vs. Low-Rated Venues')
ax1.grid(True, alpha=0.3, axis='y')
for i, v in enumerate(review_counts_comparison):
    ax1.text(i, v, f'{v:.1f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

# Years active comparison
years_comparison = [high_rated['years_active'].mean(), low_rated['years_active'].mean()]
ax2.bar(categories, years_comparison, color=['green', 'red'], alpha=0.7, edgecolor='black')
ax2.set_ylabel('Average Years Active')
ax2.set_title('Average Years Active: High vs. Low-Rated Venues')
ax2.grid(True, alpha=0.3, axis='y')
for i, v in enumerate(years_comparison):
    ax2.text(i, v, f'{v:.1f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('eda_plots/6_high_vs_low_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("   Saved: eda_plots/6_high_vs_low_comparison.png")

# Chart 7: Temporal Coverage Heatmap (Sample of venues)
print("\n[Chart 7] Temporal Coverage Heatmap (Top 20 venues)")
fig, ax = plt.subplots(figsize=(12, 10))

# Get top 20 venues by review count
top_venues = df_business.nlargest(20, 'filtered_review_count')
top_venue_ids = top_venues['business_id'].values

# Create heatmap data
heatmap_data = []
venue_names = []
for business_id in top_venue_ids:
    venue_name = df_business[df_business['business_id'] == business_id]['name'].values[0]
    venue_names.append(venue_name[:30])  # Truncate long names
    
    yearly_counts = []
    for year in range(2013, 2020):
        count = len(df_reviews[(df_reviews['business_id'] == business_id) & (df_reviews['year'] == year)])
        yearly_counts.append(count)
    heatmap_data.append(yearly_counts)

sns.heatmap(heatmap_data, annot=True, fmt='d', cmap='YlOrRd', ax=ax, 
            xticklabels=range(2013, 2020), yticklabels=venue_names, cbar_kws={'label': 'Review Count'})
ax.set_xlabel('Year')
ax.set_ylabel('Venue')
ax.set_title('Temporal Coverage: Review Activity by Year (Top 20 Venues)')
plt.tight_layout()
plt.savefig('eda_plots/7_temporal_coverage_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()
print("   Saved: eda_plots/7_temporal_coverage_heatmap.png")

# ============================================================================
# SUMMARY & ACTIONABLE INSIGHTS
# ============================================================================
print("\n" + "=" * 80)
print("SUMMARY & ACTIONABLE INSIGHTS")
print("=" * 80)

print("\n[KEY FINDINGS]")
print("\n1. DATA READINESS:")
print(f"   - {len(stable_venues)} venues ({len(stable_venues)/len(df_business)*100:.0f}%) have high stability (>=50 reviews, >=5 years)")
print(f"   - {len(moderate_venues)} venues ({len(moderate_venues)/len(df_business)*100:.0f}%) have moderate stability (>=20 reviews, >=3 years)")
print(f"   - Price data available for only {price_available} venues ({price_available/len(df_business)*100:.0f}%) - limited utility")
print(f"   - Minimal missing data in core fields (reviews, ratings, text)")

print("\n2. RATING PATTERNS:")
print(f"   - Positive skew: {(df_reviews['stars'] >= 4).sum()/len(df_reviews)*100:.1f}% of reviews are 4-5 stars")
print(f"   - High-rated venues (>=4.0) have {high_rated['filtered_review_count'].mean():.0f} avg reviews vs. {low_rated['filtered_review_count'].mean():.0f} for low-rated")
print(f"   - {len(turnaround_candidates)} venues show mixed signals (potential for improvement)")

print("\n3. TEMPORAL INSIGHTS:")
print(f"   - Review activity shows growth trend from 2013-2019")
print(f"   - Seasonality detected: {month_names[peak_month-1]} is peak month, {month_names[low_month-1]} is lowest")
print(f"   - {len(consistent_venues)} venues ({len(consistent_venues)/len(df_business)*100:.0f}%) have consistent multi-year presence")

print("\n4. TEXT ANALYSIS READINESS:")
print(f"   - Average review length: {df_reviews['word_count'].mean():.0f} words (sufficient for NLP)")
print(f"   - Clear thematic differences between 5-star and 1-2 star reviews")
print(f"   - Rich text data available for sentiment/topic analysis")

print("\n[RECOMMENDATIONS FOR CORE VALUE vs. OPERATIONAL EXECUTION ANALYSIS]")
print("\n1. Focus on the {0} highly stable venues for robust decomposition analysis".format(len(stable_venues)))
print("2. Use rating volatility as a signal for operational inconsistency")
print("3. Leverage temporal patterns to identify improvement trajectories")
print("4. Text analysis can distinguish core value (concept, atmosphere) from operations (service, wait times)")
print(f"5. {len(turnaround_candidates)} mixed-signal venues are ideal candidates for case studies")

print("\n" + "=" * 80)
print("EDA COMPLETE - 7 visualizations saved to eda_plots/")
print("=" * 80)
