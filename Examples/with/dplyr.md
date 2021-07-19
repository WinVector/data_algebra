sql\_example
================

Just a peak at what `dbplyr` generated `SQL` looks like.

``` r
library(dplyr)
```

    ## 
    ## Attaching package: 'dplyr'

    ## The following objects are masked from 'package:stats':
    ## 
    ##     filter, lag

    ## The following objects are masked from 'package:base':
    ## 
    ##     intersect, setdiff, setequal, union

``` r
library(dbplyr)
```

    ## 
    ## Attaching package: 'dbplyr'

    ## The following objects are masked from 'package:dplyr':
    ## 
    ##     ident, sql

``` r
library(DBI)

connection <- DBI::dbConnect(RSQLite::SQLite(), ":memory:")
```

``` r
d = data.frame(
  x = c(1, 2, 3)
)

d_db = copy_to(connection, d)
```

``` r
ops <- d_db %>%
  mutate(z = x + 1) %>%
  mutate(q = x + 3) %>%
  mutate(h = x + 6)

ops %>% show_query()
```

    ## <SQL>
    ## SELECT `x`, `z`, `q`, `x` + 6.0 AS `h`
    ## FROM (SELECT `x`, `z`, `x` + 3.0 AS `q`
    ## FROM (SELECT `x`, `x` + 1.0 AS `z`
    ## FROM `d`))

``` r
ops <- d_db %>%
  mutate(z = x + 1) %>%
  mutate(q = z + 2) %>%
  mutate(h = q + 3)

ops %>% show_query()
```

    ## <SQL>
    ## SELECT `x`, `z`, `q`, `q` + 3.0 AS `h`
    ## FROM (SELECT `x`, `z`, `z` + 2.0 AS `q`
    ## FROM (SELECT `x`, `x` + 1.0 AS `z`
    ## FROM `d`))

``` r
dbDisconnect(connection)
```
