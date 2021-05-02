- Refer to SQL course of codecademy
##### Introduction to SQL
- SQL style guide: https://www.sqlstyle.guide/
- We can not only compare a column with a value, but we can also compare columns to other columns.\: `WHERE Men < Women`
- Four types of SQL commands:
    - Data query language
    - Data definition language
    - Data control language
    - Data manipulation language
- [Terms-in-sql.jpg]

##### Summary and group summary Statistics
- Different to Aggregate functions, functions that, when we pass them a column as input, return (a transformation of the input) another column: LENGTH(), LOWER(), || operator for concatenate strings 
- If we try to divide two integer columns, SQLite (and most other SQL dialects) will round down and return integer values
- To get float value, we can use the CAST() function to the transform the columns into Float type:
```sql
SELECT CAST(Women AS Float) / CAST(Total AS Float) AS women_ratio
  FROM new_grads 
 LIMIT 5;
```

##### Subqueries
- A subquery is a query nested within another query.
```sql
-- Write a SQL statement that computes the proportion (as a float value) of rows that contain above-average values for ShareWomen
SELECT CAST(COUNT(*) AS Float)/CAST((SELECT COUNT(*) 
                                     FROM recent_grads) 
                                    AS Float) AS proportion_abv_avg
  FROM recent_grads
 WHERE ShareWomen > (SELECT AVG(ShareWomen)
                       FROM recent_grads
                    );
```
- [Sql-operators.jpg]
```sql
-- create a query that displays the Major and Major_category columns, for the rows where Major_category is one of the three highest group level sums for the Total column.
SELECT Major_category, Major
  FROM recent_grads
 WHERE Major_category IN (SELECT Major_category
                         FROM recent_grads
                         GROUP BY Major_category
                          ORDER BY SUM(TOTAL) DESC
                         LIMIT 3);
```
```sql
-- Filters for the rows where ratio is greater than avg_ratio
SELECT Major, Major_category, CAST(Sample_size AS Float)/Total AS ratio 
FROM recent_grads
WHERE ratio > (SELECT AVG(CAST(Sample_size AS FLOAT)/Total) AS avg_ratio FROM recent_grads);  -- error when using SELECT AVG(ratio) 
```
