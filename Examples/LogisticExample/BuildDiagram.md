Build Diagram
================

``` r
library(yaml)
library(wrapr)
library(rquery)
library(rqdatatable)

r_yaml <- yaml.load_file("pipeline_yaml.txt")
r_ops <- convert_yaml_to_pipeline(r_yaml)
cat(format(r_ops))
```

    ## mk_td("d", c(
    ##   "subjectID",
    ##   "surveyCategory",
    ##   "assessmentTotal",
    ##   "irrelevantCol1",
    ##   "irrelevantCol2")) %.>%
    ##  extend(.,
    ##   probability %:=% exp(assessmentTotal * 0.237)) %.>%
    ##  extend(.,
    ##   total %:=% sum(probability),
    ##   partitionby = c('subjectID'),
    ##   orderby = c(),
    ##   reverse = c()) %.>%
    ##  extend(.,
    ##   probability %:=% probability / total) %.>%
    ##  extend(.,
    ##   sort_key %:=% -(probability)) %.>%
    ##  extend(.,
    ##   row_number %:=% row_number(),
    ##   partitionby = c('subjectID'),
    ##   orderby = c('sort_key'),
    ##   reverse = c()) %.>%
    ##  select_rows(.,
    ##    row_number == 1) %.>%
    ##  select_columns(., c(
    ##    "subjectID", "surveyCategory", "probability")) %.>%
    ##  rename_columns(.,
    ##   c('diagnosis' = 'surveyCategory'))

``` r
r_ops %.>%
  op_diagram(.) %.>% 
  DiagrammeR::grViz(.)
```

![](BuildDiagram_files/figure-gfm/diagram-1.png)<!-- -->

See also <https://github.com/WinVector/rquery/tree/master/Examples/yaml>
.
