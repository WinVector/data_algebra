Module data_algebra.data_model_space
====================================

Classes
-------

`DataModelSpace(data_model: Optional[data_algebra.data_model.DataModel] = None)`
:   A data space as a map of mapped data_model data frames.
    
    Build an isolated execution space. Good for enforcing different data model adaptors or alternatives.
    
    :param data_model: execution model

    ### Ancestors (in MRO)

    * data_algebra.data_space.DataSpace
    * abc.ABC

    ### Methods

    `close(self) ‑> None`
    :