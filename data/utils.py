import pandas as pd
import numpy as np
import glob
import os

class ExcelDataLoader():

    def __init__(
            self,
            path,
            indexName: Optional[ str ] = '案號',
            header: int = 1,
            dropNaRowName: Optional[ str ] = None,
            fillNaWith: Optional[ Union[ int, float, str ] ] = 0 ) -> None:
        self.df = pd.read_excel( path, header=header )
        self.index_name = indexName
        self.cols = self.df.columns
        if dropNaRowName:
            self.df = self.df.drop( np.where( self.df[ dropNaRowName ].isna() )[ 0 ] )
        if fillNaWith is not False and fillNaWith is not None:
            self.df = self.df.fillna( fillNaWith )

    def __getitem__( self, index ):
        if isinstance( index, Iterable ) and sum( [ isinstance( i, ( Number, str ) ) for i in index ] ) == 2:
            i, j = [ i for i in index if isinstance( i, ( Number, str ) ) ]
            if self.index_name is None or isinstance( i, int ):
                if not isinstance( i, int ):
                    raise ValueError( f"indexName is {self.index_name}, but first idx {i=} not a integer" )
                return self.df.loc[ i ][ j ]
            else:
                return self.df[ self.df[ self.index_name ] == i ][ j ].item()
        elif isinstance( index, str ):
            return self.df[ index ]