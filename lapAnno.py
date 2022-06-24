# %%
import pylightxl as xl


# %%
def read_contour_txt( recordpath: str, sept: str = ' ' ):
    recorddata = {}
    tempdata = []
    with open( recordpath, 'r' ) as f:
        for line in f.readlines():
            words = [ word.replace( '\n', '' ) for word in line.split( sept ) if word and word != '\n' ]
            # print( words )
            if not words:
                continue
            if words[ 0 ].startswith( 'HNNLAP' ):
                tempdata = recorddata[ words[ 0 ].replace( 'CT', '' )[ 7: ] ] = {}
            elif words[ 0 ].startswith( 'LAP' ):
                tempdata[ words[ 0 ] ] = [ 1 if i == 'pos' else 0 for i in words[ 1: ] ]
    return recorddata


# %%
if __name__ == '__main__':
    recordpath = '../../../0826-LAP-shareData/20220101-中榮病人/contour.txt'
    recorddata = read_contour_txt( recordpath, ' ' )
    print( recorddata )


# %%
def read_contour_execl( recordpath: str ):
    recorddata = {}
    tempdata = []
    db = xl.readxl( recordpath )
    ws = db.ws( ws=db.ws_names[ 0 ] )
    for i, row in enumerate( ws.rows ):
        if row[ 1 ].startswith( 'HNNLAP' ):
            tempdata = recorddata[ row[ 1 ].replace( 'CT', '' )[ 7: ] ] = {}
        elif row[ 1 ].startswith( 'LAP' ):
            tempdata[ row[ 1 ] ] = [ 1 if 'pos' in row[ i ] else 0 for i in range( 2, 3 + 2 ) ]
        # print( row )
    return recorddata


# %%
if __name__ == '__main__':
    recordpath = '../../../0826-LAP-shareData/20210128-中榮病人-121例/LAP_detail_0524.xlsx'
    recorddata = read_contour_execl( recordpath )
    recorddata


# %%
def read_contour_execl_2018( recordpath: str ):
    recorddata = {}
    db = xl.readxl( recordpath )
    ws = db.ws( ws=db.ws_names[ 2 ] )
    # print( db.ws_names[ 2 ] )
    for i, row in enumerate( ws.rows ):
        # print( row )
        if i == 0:
            continue
        if row[ 0 ] not in recorddata:
            recorddata[ row[ 0 ] ] = { f'LAP{row[1]}': [ 1 if '+' in row[ i ] else 0 for i in range( 2, 3 + 2 ) ]}
        else:
            recorddata[ row[ 0 ] ][ f'LAP{row[1]}' ] = [ 1 if '+' in row[ i ] else 0 for i in range( 2, 3 + 2 ) ]
    return recorddata


# %%
if __name__ == '__main__':
    recordpath = '../../../0826-LAP-shareData/20220301-HNC-2018/2018 hnc.xls 術前CT日期.xls ~0408.xlsx'
    recorddata = read_contour_execl_2018( recordpath )
    print( recorddata )
# %%
