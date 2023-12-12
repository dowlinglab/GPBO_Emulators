import sys
import numpy as np
import pandas as pd
from datetime import datetime
from scipy.stats import qmc
import itertools
from itertools import combinations_with_replacement, combinations, permutations

import bo_methods_lib
import pytest
from bo_methods_lib.GPBO_Classes_New import * #Fix this later
from bo_methods_lib.GPBO_Class_fxns import * #Fix this later

#Set Date and Time
dateTimeObj = datetime.now()
timestampStr = dateTimeObj.strftime("%d-%b-%Y (%H:%M:%S)")
DateTime_str = dateTimeObj.strftime("%Y/%m/%d/%H-%M")
DateTime_none = None ##For Testing

def test_bo_methods_lib_imported():
    """Sample test, will always pass so long as import statement worked."""
    assert "bo_methods_lib" in sys.modules

#This test function tests whether Enum Classes will call the correct errors
            ##function, enum_value
Enum_array = [[Method_name_enum, -1],
              [Method_name_enum, 0],
              [Method_name_enum, 7],
              [Method_name_enum, "string"],
              [Kernel_enum, -1],
              [Kernel_enum, 0],
              [Kernel_enum, 4],
              [Kernel_enum, "string"],
              [Gen_meth_enum, -1],
              [Gen_meth_enum, 0],
              [Gen_meth_enum, 3],
              [Gen_meth_enum, "string"],
              [Obj_enum, -1],
              [Obj_enum, 0],
              [Obj_enum, 3],
              [Obj_enum, "string"],
              [CS_name_enum, -1],
              [CS_name_enum, 0],
              [CS_name_enum, 10],
              [CS_name_enum, "string"],
              [Ep_enum, -1],
              [Ep_enum, 0],
              [Ep_enum, 5],
              [Ep_enum, "string"]]
@pytest.mark.parametrize("function, enum_value", Enum_array)
def test_CaseStudyParameters(function, enum_value):
    with pytest.raises(ValueError):
        value = function(enum_value)   