Module data_algebra.arrow
=========================

Functions
---------

    
`fmt_as_arrow(ops) ‑> str`
:   

Classes
-------

`Arrow()`
:   Arrow from category theory: see Steve Awody,
    "Category Theory, 2nd Edition", Oxford Univ. Press, 2010 pg. 4.
    Essentially this is a tool to expose associativity, without forcing
    things to be a function to expose this effect.

    ### Ancestors (in MRO)

    * data_algebra.shift_pipe_action.ShiftPipeAction
    * abc.ABC

    ### Descendants

    * data_algebra.arrow.DataOpArrow

    ### Methods

    `act_on(self, b)`
    :   act on b, must associate with composition

    `cod(self)`
    :   return co-domain, object at head of arrow

    `dom(self)`
    :   return domain, object at base of arrow

    `transform(self, X)`
    :   transform X, may or may not associate with composition

`DataOpArrow(pipeline, *, free_table_key=None)`
:   Represent a dag of operators as a categorical arrow.

    ### Ancestors (in MRO)

    * data_algebra.arrow.Arrow
    * data_algebra.shift_pipe_action.ShiftPipeAction
    * abc.ABC

    ### Methods

    `act_on(self, b, *, correct_ordered_first_call: bool = False)`
    :   Apply self onto b.
        
        :param b: item to act on, or item that has been sent to self.
        :param correct_ordered_first_call: if True indicates this call is from __rshift__ or __rrshift__ and not the fallback paths.

    `cod_as_table(self)`
    :

    `dom_as_table(self)`
    :

    `format_end_description(self, *, required_cols, align_right=70, sep_width=2)`
    :

    `get_feature_names(self)`
    :

    `required_columns(self)`
    :