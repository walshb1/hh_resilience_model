import ogr,csv,sys
import pandas as pd

def qgis_to_csv(shpfile=None,csvfile=None):
    if shpfile is None: 
        shpfile=r'../inputs/__haz_data_GLOFRIS/flood_risk_maps_and_data/flood_risk_data_by_state/aqueduct_global_flood_risk_data_by_state_20150304.shp' #sys.argv[1]
    if csvfile is None:
        csvfile=r'../inputs/__haz_data_GLOFRIS/flood_risk_maps_and_data/flood_risk_data_by_state/aqueduct_global_flood_risk_data_by_state_20150304.csv' #sys.argv[2]
    
    #Open files
    csvfile=open(csvfile,'w')
    ds=ogr.Open(shpfile)
    lyr=ds.GetLayer()
    
    #Get field names
    dfn=lyr.GetLayerDefn()
    nfields=dfn.GetFieldCount()
    fields=[]
    for i in range(nfields):
        fields.append(dfn.GetFieldDefn(i).GetName())
    fields.append('kmlgeometry')
    csvwriter = csv.DictWriter(csvfile, fields)
    try:csvwriter.writeheader() #python 2.7+
    except:csvfile.write(','.join(fields)+'\n')

    # Write attributes and kml out to csv
    for feat in lyr:
        attributes=feat.items()
        geom=feat.GetGeometryRef()
        attributes['kmlgeometry']=geom.ExportToKML()
        csvwriter.writerow(attributes)

    #clean up
    del csvwriter,lyr,ds
    csvfile.close()

def choose_country(pais=['XX','XX']):

    try: df = pd.read_csv('../inputs/'+pais[1]+'/flood_risk_by_state.csv').set_index('woe_name')
    except: 
        df = pd.read_csv('../inputs/__haz_data_GLOFRIS/flood_risk_maps_and_data/flood_risk_data_by_state/aqueduct_global_flood_risk_data_by_state_20150304.csv').set_index('woe_name')
        df = df.loc[df['admin']==pais[0]]
    #
    try: df = df.drop([_c for _c in ['admin','unit_id','unit_name','kmlgeometry'] if _c in df.columns],axis=1)
    except: pass
    df.to_csv('../inputs/'+pais[1]+'/flood_risk_by_state.csv')
    #
    #try:
    for _code in ['G','P','U']:
        _df = df[[_c for _c in df.columns if _code in _c and _code+'30' not in _c]]
        _df = _df.rename(columns={_code+'10_bh_2':2,
                                  _code+'10_bh_5':5,
                                  _code+'10_bh_10':10,
                                  _code+'10_bh_25':25,
                                  _code+'10_bh_50':50,
                                  _code+'10_bh_100':100,
                                  _code+'10_bh_250':250,
                                  _code+'10_bh_500':500,
                                  _code+'10_bh_1T':1000}).stack().to_frame(name={'G':'gdp_affected','P':'pop_affected','U':'urban_losses'}[_code])
        _df.index.names = ['department','rp']

        try: df_out = pd.concat([df_out,_df],axis=1)
        except: df_out = _df.copy()
        
    df_out.to_csv('../inputs/'+pais[1]+'/flood_risk_by_state.csv')
    return True

#choose_country(['Bolivia','BO'])
