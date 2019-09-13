
expr_map_to_expr_array <- function(emap) {
  r = vapply(names(emap), function(k) paste0(k, " := ", emap[[k]]), character(1))
  names(r) <- NULL
  return(r)
}

convert_yaml_to_pipeline <- function(rep, source=NULL) {
  if(is.null(names(rep))) {
    # unnamed list, a pipeline
    res <- convert_yaml_to_pipeline(rep[[1]])
    for(i in seqi(2, length(rep))) {
      res <- convert_yaml_to_pipeline(rep[[i]], source=res)
    }
    return(res)
  } else {
    # named list, a stage
    op = rep$op
    if(op=="TableDescription") {
      return(mk_td(table_name = rep$table_name,
                   columns = rep$column_names))
    } else if(op=="Extend") {
      return(extend_se(source,
                       assignments = expr_map_to_expr_array(rep$ops),
                       partitionby = as.character(rep$partition_by),
                       orderby = as.character(rep$order_by),
                       reverse = as.character(rep$reverse)))
    } else if(op=="SelectRows") {
      return(select_rows_se(source,
                            expr = rep$expr))
    } else if(op=="SelectColumns") {
      return(select_columns(source, columns=rep$columns))
    } else if(op=="Rename") {
      return(rename_columns(source, cmap=rep$column_remapping))  # TODO: see if map is dumped (or __repr__() call wrote a string)
    } else if(op=="Order") {
      return(orderby(source, cols=rep$order_columns, reverse=rep$reverse, limit=rep$limit))
    } else {
      stop("Unexpected node type: " + op)
    }
  }
}
