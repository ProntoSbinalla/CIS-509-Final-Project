"""
Filter Yelp Dataset for Nashville Music Venues (2013-2019)

This script:
1. Loads business.json and filters for Nashville businesses in "Music Venues" category
2. Extracts business_ids for these venues
3. Loads review.json and filters for matching reviews from 2013-2019
4. Applies minimum threshold of 20 reviews per venue
5. Saves filtered dataset and generates summary statistics
"""

import json
import pandas as pd
from datetime import datetime
from collections import defaultdict
import os

print("=" * 80)
print("YELP DATASET FILTERING: Nashville Music Venues (2013-2019)")
print("=" * 80)

# File paths
BUSINESS_FILE = "yelp_academic_dataset_business.json"
REVIEW_FILE = "yelp_academic_dataset_review.json"
OUTPUT_DIR = "filtered_data"

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Filtering criteria
TARGET_CITY = "Nashville"
TARGET_CATEGORY = "Music Venues"
START_DATE = "2013-01-01"
END_DATE = "2019-12-31"
MIN_REVIEWS = 20

print(f"\n[*] Filtering Criteria:")
print(f"   City: {TARGET_CITY}")
print(f"   Category: {TARGET_CATEGORY}")
print(f"   Date Range: {START_DATE} to {END_DATE}")
print(f"   Minimum Reviews: {MIN_REVIEWS}")
print()

# ============================================================================
# STEP 1: Load and filter businesses
# ============================================================================
print("[1] STEP 1: Loading and filtering businesses...")

nashville_music_venues = []
business_count = 0

with open(BUSINESS_FILE, 'r', encoding='utf-8') as f:
    for line in f:
        business_count += 1
        if business_count % 10000 == 0:
            print(f"   Processed {business_count:,} businesses...", end='\r')
        
        business = json.loads(line)
        
        # Check if business is in Nashville
        if business.get('city', '').lower() == TARGET_CITY.lower():
            # Check if business has "Music Venues" in categories
            categories = business.get('categories', '')
            if categories and TARGET_CATEGORY in categories:
                nashville_music_venues.append(business)

print(f"   Processed {business_count:,} businesses total.")
print(f"[+] Found {len(nashville_music_venues)} Nashville Music Venues")

# Extract business IDs
venue_ids = {venue['business_id'] for venue in nashville_music_venues}
print(f"   Extracted {len(venue_ids)} unique business IDs")

# Create business lookup dictionary
business_lookup = {venue['business_id']: venue for venue in nashville_music_venues}

# ============================================================================
# STEP 2: Load and filter reviews
# ============================================================================
print(f"\n[2] STEP 2: Loading and filtering reviews (this may take a while for 5GB file)...")

filtered_reviews = []
review_count = 0
matching_reviews = 0
review_counts_by_business = defaultdict(int)

# Parse date range
start_dt = datetime.strptime(START_DATE, "%Y-%m-%d")
end_dt = datetime.strptime(END_DATE, "%Y-%m-%d")

with open(REVIEW_FILE, 'r', encoding='utf-8') as f:
    for line in f:
        review_count += 1
        if review_count % 100000 == 0:
            print(f"   Processed {review_count:,} reviews, found {matching_reviews:,} matches...", end='\r')
        
        review = json.loads(line)
        
        # Check if review is for one of our Nashville music venues
        if review['business_id'] in venue_ids:
            # Check if review is within date range
            review_date = datetime.strptime(review['date'], "%Y-%m-%d %H:%M:%S")
            if start_dt <= review_date <= end_dt:
                filtered_reviews.append(review)
                review_counts_by_business[review['business_id']] += 1
                matching_reviews += 1

print(f"   Processed {review_count:,} reviews total.")
print(f"[+] Found {matching_reviews:,} reviews matching criteria")

# ============================================================================
# STEP 3: Apply minimum review threshold
# ============================================================================
print(f"\n[3] STEP 3: Applying minimum review threshold ({MIN_REVIEWS} reviews)...")

# Filter businesses that have at least MIN_REVIEWS reviews
qualifying_business_ids = {
    business_id for business_id, count in review_counts_by_business.items()
    if count >= MIN_REVIEWS
}

print(f"   Businesses before threshold: {len(venue_ids)}")
print(f"   Businesses after threshold: {len(qualifying_business_ids)}")
print(f"   Removed: {len(venue_ids) - len(qualifying_business_ids)} venues")

# Filter reviews to only include qualifying businesses
final_reviews = [
    review for review in filtered_reviews
    if review['business_id'] in qualifying_business_ids
]

# Filter businesses to only include qualifying ones
final_businesses = [
    business for business in nashville_music_venues
    if business['business_id'] in qualifying_business_ids
]

print(f"[+] Final dataset: {len(final_businesses)} venues, {len(final_reviews)} reviews")

# ============================================================================
# STEP 4: Save filtered datasets
# ============================================================================
print(f"\n[4] STEP 4: Saving filtered datasets...")

# Save businesses
business_output = os.path.join(OUTPUT_DIR, "nashville_music_venues.json")
with open(business_output, 'w', encoding='utf-8') as f:
    json.dump(final_businesses, f, indent=2)
print(f"   [+] Saved businesses to: {business_output}")

# Save reviews
reviews_output = os.path.join(OUTPUT_DIR, "nashville_music_venue_reviews.json")
with open(reviews_output, 'w', encoding='utf-8') as f:
    json.dump(final_reviews, f, indent=2)
print(f"   [+] Saved reviews to: {reviews_output}")

# Create combined dataset with business info + reviews
combined_data = []
for business in final_businesses:
    business_id = business['business_id']
    business_reviews = [r for r in final_reviews if r['business_id'] == business_id]
    
    combined_data.append({
        'business_info': {
            'business_id': business['business_id'],
            'name': business['name'],
            'address': business.get('address', ''),
            'city': business.get('city', ''),
            'state': business.get('state', ''),
            'postal_code': business.get('postal_code', ''),
            'latitude': business.get('latitude'),
            'longitude': business.get('longitude'),
            'stars': business.get('stars'),
            'review_count': business.get('review_count'),
            'is_open': business.get('is_open'),
            'attributes': business.get('attributes', {}),
            'categories': business.get('categories', ''),
            'hours': business.get('hours', {})
        },
        'reviews': business_reviews,
        'filtered_review_count': len(business_reviews)
    })

combined_output = os.path.join(OUTPUT_DIR, "combined_nashville_music_venues.json")
with open(combined_output, 'w', encoding='utf-8') as f:
    json.dump(combined_data, f, indent=2)
print(f"   [+] Saved combined dataset to: {combined_output}")

# ============================================================================
# STEP 5: Generate summary statistics
# ============================================================================
print(f"\n[5] STEP 5: Generating summary statistics...")

# Convert reviews to DataFrame for easier analysis
reviews_df = pd.DataFrame(final_reviews)
businesses_df = pd.DataFrame(final_businesses)

# Date range coverage
reviews_df['date'] = pd.to_datetime(reviews_df['date'])
min_date = reviews_df['date'].min()
max_date = reviews_df['date'].max()

# Reviews per venue
reviews_per_venue = reviews_df.groupby('business_id').size()

# Star rating distribution
star_distribution = reviews_df['stars'].value_counts().sort_index()

# Summary statistics
summary_stats = {
    'filtering_criteria': {
        'city': TARGET_CITY,
        'category': TARGET_CATEGORY,
        'date_range': f"{START_DATE} to {END_DATE}",
        'minimum_reviews': MIN_REVIEWS
    },
    'dataset_summary': {
        'total_venues': len(final_businesses),
        'total_reviews': len(final_reviews),
        'actual_date_range': {
            'earliest_review': str(min_date),
            'latest_review': str(max_date)
        },
        'reviews_per_venue': {
            'mean': float(reviews_per_venue.mean()),
            'median': float(reviews_per_venue.median()),
            'min': int(reviews_per_venue.min()),
            'max': int(reviews_per_venue.max()),
            'std': float(reviews_per_venue.std())
        },
        'star_rating_distribution': {
            str(int(k)): int(v) for k, v in star_distribution.items()
        },
        'average_star_rating': float(reviews_df['stars'].mean())
    },
    'venue_details': [
        {
            'name': venue['name'],
            'business_id': venue['business_id'],
            'overall_stars': venue.get('stars'),
            'total_review_count': venue.get('review_count'),
            'filtered_review_count': review_counts_by_business[venue['business_id']],
            'address': venue.get('address', '')
        }
        for venue in final_businesses
    ]
}

# Save summary statistics
summary_output = os.path.join(OUTPUT_DIR, "summary_statistics.json")
with open(summary_output, 'w', encoding='utf-8') as f:
    json.dump(summary_stats, f, indent=2)
print(f"   [+] Saved summary statistics to: {summary_output}")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("[SUCCESS] FILTERING COMPLETE!")
print("=" * 80)
print(f"\n[*] SUMMARY STATISTICS:")
print(f"   Total Venues: {len(final_businesses)}")
print(f"   Total Reviews: {len(final_reviews)}")
print(f"   Date Range: {min_date.date()} to {max_date.date()}")
print(f"\n   Reviews per Venue:")
print(f"      Mean: {reviews_per_venue.mean():.1f}")
print(f"      Median: {reviews_per_venue.median():.1f}")
print(f"      Range: {reviews_per_venue.min()} - {reviews_per_venue.max()}")
print(f"\n   Average Star Rating: {reviews_df['stars'].mean():.2f}")
print(f"\n   Star Distribution:")
for stars, count in star_distribution.items():
    print(f"      {int(stars)} stars: {count} reviews ({count/len(final_reviews)*100:.1f}%)")

print(f"\n[*] Output Files:")
print(f"   - {business_output}")
print(f"   - {reviews_output}")
print(f"   - {combined_output}")
print(f"   - {summary_output}")

print("\n" + "=" * 80)
print("[*] Nashville Music Venues Dataset Ready for Analysis!")
print("=" * 80)
