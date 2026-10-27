# EDA Findings

Charts are in `reports/eda/`.

**Q1. Distribution of property prices**  
Prices range 10-500 lakhs, mean 255, median 254; roughly uniform.

**Q2. Distribution of property sizes**  
Sizes range 500-5000 sq ft, median 2747.

**Q3. Price per sq ft by property type**  
Median price/sq ft by type: {'Apartment': 9245.0, 'Independent House': 9305.0, 'Villa': 9190.0}.

**Q4. Property size vs price**  
Correlation size-price = -0.003 (near zero, size alone does not drive price).

**Q5. Outliers in price per sq ft and size**  
Price/sq ft is right-skewed with many high-end outliers (capped at the 1st-99th percentile in preprocessing); size has no outliers.

**Q6. Average price per sq ft by state**  
Highest: Karnataka (13150); lowest: Rajasthan (12829).

**Q7. Average property price by city**  
Highest: Bangalore (258.5 L); lowest: Cuttack (250.8 L). Spread is narrow.

**Q8. Median property age by locality (distribution over 500 localities)**  
Locality median ages range 16-21 years; typical locality median 18 years.

**Q9. BHK distribution across top 10 cities**  
BHK 1-5 is evenly spread in every city.

**Q10. Price trend (by year built) for top 5 most expensive localities**  
Top localities: Locality_395, Locality_366, Locality_497, Locality_360, Locality_203. Prices fluctuate by build year with no clear upward trend.

**Q11. Correlation between numeric features**  
Raw features are almost uncorrelated with each other; Future_Price_5Y is driven by current price (~0.99).

**Q12. Nearby Schools vs price per sq ft**  
Avg price/sq ft varies only 12796-13174 across counts; weak relationship.

**Q13. Nearby Hospitals vs price per sq ft**  
Avg price/sq ft varies only 12817-13152 across counts; weak relationship.

**Q14. Price by furnished status**  
Median price: {'Furnished': 253.6, 'Semi-furnished': 253.8, 'Unfurnished': 254.2}.

**Q15. Price per sq ft by facing direction**  
Median: {'East': 9234.0, 'North': 9219.0, 'South': 9225.0, 'West': 9308.0}.

**Q16. Properties by owner type**  
{'Broker': 83479, 'Owner': 83268, 'Builder': 83253}.

**Q17. Properties by availability status**  
{'Under_Construction': 125035, 'Ready_to_Move': 124965}.

**Q18. Parking space vs price**  
Median price: {'No': 253.4, 'Yes': 254.4}; no meaningful difference.

**Q19. Amenities vs price per sq ft**  
Avg price/sq ft by amenity count: {1: 12928.0, 2: 12956.0, 3: 12942.0, 4: 12970.0, 5: 12998.0}.

**Q20. Public transport vs price per sq ft and investment potential**  
Price/sq ft barely changes, but Good Investment share rises with access: {'Low': 0.483, 'Medium': 0.724, 'High': 0.872}.
