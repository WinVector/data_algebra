Module data_algebra.data_schema
===============================
Tools for checking incoming and outgoing names and types of functions of data frames

Functions
---------

    
`non_null_types_in_frame(d: pandas.core.frame.DataFrame) ‑> Dict[str, Optional[Set[Type]]]`
:   Return dictionary of non-null types seen in dataframe columns.

Classes
-------

`SchemaBase(arg_specs: Optional[Dict[str, Any]] = None, *, return_spec=None)`
:   Input and output schema decorator.
    
    Pandas data frames must have at least declared columns and no unexpected types in columns.
    Nulls/Nones/NaNs values are not considered to have type (treating them as missingness).
    None as type constraints are considered no-type (unfailable).
    
    :param arg_specs: dictionary of named args to type specifications.
    :param return_spec: optional return type specification.

    ### Descendants

    * data_algebra.data_schema.SchemaMock
    * data_algebra.data_schema.SchemaRaises

`SchemaCheckSwitch()`
:   From: https://python-patterns.guide/gang-of-four/singleton/

    ### Methods

    `is_on(self) ‑> bool`
    :

    `off(self) ‑> None`
    :

    `on(self) ‑> None`
    :

`SchemaMock(arg_specs: Optional[Dict[str, Any]] = None, *, return_spec=None)`
:   Build schema, but do not enforce or attach to fn
    
    Pandas data frames must have at least declared columns and no unexpected types in columns.
    Nulls/Nones/NaNs values are not considered to have type (treating them as missingness).
    None as type constraints are considered no-type (unfailable).
    
    :param arg_specs: dictionary of named args to type specifications.
    :param return_spec: optional return type specification.

    ### Ancestors (in MRO)

    * data_algebra.data_schema.SchemaBase

`SchemaRaises(arg_specs: Optional[Dict[str, Any]] = None, *, return_spec=None)`
:   Input and output schema decorator.
    Raises TypeError on schema violations.
    
    Pandas data frames must have at least declared columns and no unexpected types in columns.
    Nulls/Nones/NaNs values are not considered to have type (treating them as missingness).
    None as type constraints are considered no-type (unfailable).
    
    :param arg_specs: dictionary of named args to type specifications.
    :param return_spec: optional return type specification.

    ### Ancestors (in MRO)

    * data_algebra.data_schema.SchemaBase

    ### Methods

    `check_args(self, *, arg_names: List[str], fname: str, args, kwargs) ‑> None`
    :

    `check_return(self, *, fname: str, return_value) ‑> None`
    :