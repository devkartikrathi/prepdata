import joblib
import pandas as pd
import numpy as np
import json
import sys

def load_model():
    """Load the trained model"""
    try:
        model = joblib.load('public/models/trained_model1.pkl')
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def preprocess_input(data):
    """Preprocess input data to match model expectations"""
    try:
        # Convert input data to DataFrame
        df = pd.DataFrame([data])
        
        # Apply log transformation to numerical features
        numerical_features = [
            'Order', 'PID', 'MS SubClass', 'Lot Frontage', 'Lot Area', 
            'Overall Qual', 'Overall Cond', 'Year Built', 'Year Remod/Add',
            'Mas Vnr Area', 'BsmtFin SF 1', 'BsmtFin SF 2', 'Bsmt Unf SF',
            'Total Bsmt SF', '1st Flr SF', '2nd Flr SF', 'Low Qual Fin SF',
            'Gr Liv Area', 'Bsmt Full Bath', 'Bsmt Half Bath', 'Full Bath',
            'Half Bath', 'Bedroom AbvGr', 'Kitchen AbvGr', 'TotRms AbvGrd',
            'Fireplaces', 'Garage Yr Blt', 'Garage Cars', 'Garage Area',
            'Wood Deck SF', 'Open Porch SF', 'Enclosed Porch', '3Ssn Porch',
            'Screen Porch', 'Pool Area', 'Misc Val', 'Mo Sold', 'Yr Sold'
        ]
        
        for feature in numerical_features:
            if feature in df.columns:
                df[feature] = np.log1p(df[feature])
        
        # Ensure all expected columns are present
        expected_columns = [
            'Order', 'PID', 'MS SubClass', 'Lot Frontage', 'Lot Area', 'Overall Qual', 'Overall Cond', 'Year Built', 'Year Remod/Add', 'Mas Vnr Area', 'BsmtFin SF 1', 'BsmtFin SF 2', 'Bsmt Unf SF', 'Total Bsmt SF', '1st Flr SF', '2nd Flr SF', 'Low Qual Fin SF', 'Gr Liv Area', 'Bsmt Full Bath', 'Bsmt Half Bath', 'Full Bath', 'Half Bath', 'Bedroom AbvGr', 'Kitchen AbvGr', 'TotRms AbvGrd', 'Fireplaces', 'Garage Yr Blt', 'Garage Cars', 'Garage Area', 'Wood Deck SF', 'Open Porch SF', 'Enclosed Porch', '3Ssn Porch', 'Screen Porch', 'Pool Area', 'Misc Val', 'Mo Sold', 'Yr Sold', 'MS Zoning_FV', 'MS Zoning_I (all)', 'MS Zoning_RH', 'MS Zoning_RL', 'MS Zoning_RM', 'Street_Pave', 'Alley_Pave', 'Alley_missing', 'Lot Shape_IR2', 'Lot Shape_IR3', 'Lot Shape_Reg', 'Land Contour_HLS', 'Land Contour_Low', 'Land Contour_Lvl', 'Utilities_NoSewr', 'Lot Config_CulDSac', 'Lot Config_FR2', 'Lot Config_FR3', 'Lot Config_Inside', 'Land Slope_Mod', 'Land Slope_Sev', 'Neighborhood_Blueste', 'Neighborhood_BrDale', 'Neighborhood_BrkSide', 'Neighborhood_ClearCr', 'Neighborhood_CollgCr', 'Neighborhood_Crawfor', 'Neighborhood_Edwards', 'Neighborhood_Gilbert', 'Neighborhood_Greens', 'Neighborhood_GrnHill', 'Neighborhood_IDOTRR', 'Neighborhood_Landmrk', 'Neighborhood_MeadowV', 'Neighborhood_Mitchel', 'Neighborhood_NAmes', 'Neighborhood_NPkVill', 'Neighborhood_NWAmes', 'Neighborhood_NoRidge', 'Neighborhood_NridgHt', 'Neighborhood_OldTown', 'Neighborhood_SWISU', 'Neighborhood_Sawyer', 'Neighborhood_SawyerW', 'Neighborhood_Somerst', 'Neighborhood_StoneBr', 'Neighborhood_Timber', 'Neighborhood_Veenker', 'Condition 1_Feedr', 'Condition 1_Norm', 'Condition 1_PosA', 'Condition 1_PosN', 'Condition 1_RRAe', 'Condition 1_RRAn', 'Condition 1_RRNe', 'Condition 1_RRNn', 'Condition 2_Feedr', 'Condition 2_Norm', 'Condition 2_PosA', 'Condition 2_PosN', 'Condition 2_RRNn', 'Bldg Type_2fmCon', 'Bldg Type_Duplex', 'Bldg Type_Twnhs', 'Bldg Type_TwnhsE', 'House Style_1.5Unf', 'House Style_1Story', 'House Style_2.5Unf', 'House Style_2Story', 'House Style_SFoyer', 'House Style_SLvl', 'Roof Style_Gable', 'Roof Style_Gambrel', 'Roof Style_Hip', 'Roof Style_Mansard', 'Roof Style_Shed', 'Roof Matl_Tar&Grv', 'Roof Matl_WdShake', 'Roof Matl_WdShngl', 'Exterior 1st_BrkComm', 'Exterior 1st_BrkFace', 'Exterior 1st_CBlock', 'Exterior 1st_CemntBd', 'Exterior 1st_HdBoard', 'Exterior 1st_ImStucc', 'Exterior 1st_MetalSd', 'Exterior 1st_Plywood', 'Exterior 1st_PreCast', 'Exterior 1st_Stucco', 'Exterior 1st_VinylSd', 'Exterior 1st_Wd Sdng', 'Exterior 1st_WdShing', 'Exterior 2nd_Brk Cmn', 'Exterior 2nd_BrkFace', 'Exterior 2nd_CBlock', 'Exterior 2nd_CmentBd', 'Exterior 2nd_HdBoard', 'Exterior 2nd_ImStucc', 'Exterior 2nd_MetalSd', 'Exterior 2nd_Other', 'Exterior 2nd_Plywood', 'Exterior 2nd_PreCast', 'Exterior 2nd_Stone', 'Exterior 2nd_Stucco', 'Exterior 2nd_VinylSd', 'Exterior 2nd_Wd Sdng', 'Exterior 2nd_Wd Shng', 'Mas Vnr Type_BrkFace', 'Mas Vnr Type_CBlock', 'Mas Vnr Type_Stone', 'Mas Vnr Type_missing', 'Exter Qual_Fa', 'Exter Qual_Gd', 'Exter Qual_TA', 'Exter Cond_Fa', 'Exter Cond_Gd', 'Exter Cond_TA', 'Foundation_CBlock', 'Foundation_PConc', 'Foundation_Slab', 'Foundation_Stone', 'Foundation_Wood', 'Bsmt Qual_Fa', 'Bsmt Qual_Gd', 'Bsmt Qual_Po', 'Bsmt Qual_TA', 'Bsmt Qual_missing', 'Bsmt Cond_Fa', 'Bsmt Cond_Gd', 'Bsmt Cond_TA', 'Bsmt Cond_missing', 'Bsmt Exposure_Gd', 'Bsmt Exposure_Mn', 'Bsmt Exposure_No', 'Bsmt Exposure_missing', 'BsmtFin Type 1_BLQ', 'BsmtFin Type 1_GLQ', 'BsmtFin Type 1_LwQ', 'BsmtFin Type 1_Rec', 'BsmtFin Type 1_Unf', 'BsmtFin Type 1_missing', 'BsmtFin Type 2_BLQ', 'BsmtFin Type 2_GLQ', 'BsmtFin Type 2_LwQ', 'BsmtFin Type 2_Rec', 'BsmtFin Type 2_Unf', 'BsmtFin Type 2_missing', 'Heating_GasA', 'Heating_GasW', 'Heating_Grav', 'Heating_Wall', 'Heating QC_Fa', 'Heating QC_Gd', 'Heating QC_TA', 'Central Air_Y', 'Electrical_FuseF', 'Electrical_FuseP', 'Electrical_SBrkr', 'Electrical_missing', 'Kitchen Qual_Fa', 'Kitchen Qual_Gd', 'Kitchen Qual_TA', 'Functional_Maj2', 'Functional_Min1', 'Functional_Min2', 'Functional_Mod', 'Functional_Typ', 'Fireplace Qu_Fa', 'Fireplace Qu_Gd', 'Fireplace Qu_Po', 'Fireplace Qu_TA', 'Fireplace Qu_missing', 'Garage Type_Attchd', 'Garage Type_Basment', 'Garage Type_BuiltIn', 'Garage Type_CarPort', 'Garage Type_Detchd', 'Garage Type_missing', 'Garage Finish_RFn', 'Garage Finish_Unf', 'Garage Finish_missing', 'Garage Qual_Fa', 'Garage Qual_Gd', 'Garage Qual_Po', 'Garage Qual_TA', 'Garage Qual_missing', 'Garage Cond_Fa', 'Garage Cond_Gd', 'Garage Cond_Po', 'Garage Cond_TA', 'Garage Cond_missing', 'Paved Drive_P', 'Paved Drive_Y', 'Fence_GdWo', 'Fence_MnPrv', 'Fence_MnWw', 'Fence_missing', 'Misc Feature_missing', 'Sale Type_CWD', 'Sale Type_Con', 'Sale Type_ConLD', 'Sale Type_ConLI', 'Sale Type_ConLw', 'Sale Type_New', 'Sale Type_Oth', 'Sale Type_WD ', 'Sale Condition_AdjLand', 'Sale Condition_Alloca', 'Sale Condition_Family', 'Sale Condition_Normal', 'Sale Condition_Partial'
        ]
        
        # Add missing columns with default values
        for col in expected_columns:
            if col not in df.columns:
                df[col] = 0
        
        # Reorder columns to match expected order
        df = df[expected_columns]
        
        return df
    except Exception as e:
        print(f"Error preprocessing data: {e}")
        return None

def get_sample_data():
    """Get sample data for testing"""
    return {
        'Order': 7.640123172695364,
        'PID': 20.623916121292147,
        'MS SubClass': 3.044522437723423,
        'Lot Frontage': 4.442651256490317,
        'Lot Area': 9.442721128642876,
        'Overall Qual': 1.9459101490553128,
        'Overall Cond': 2.079441541679836,
        'Year Built': 7.576097340623111,
        'Year Remod/Add': 7.601901959875166,
        'Mas Vnr Area': 0.0,
        'BsmtFin SF 1': 6.169610732491456,
        'BsmtFin SF 2': 0.0,
        'Bsmt Unf SF': 6.587550014824796,
        'Total Bsmt SF': 7.0925737159746784,
        '1st Flr SF': 7.677400430514807,
        '2nd Flr SF': 0.0,
        'Low Qual Fin SF': 0.0,
        'Gr Liv Area': 7.677400430514807,
        'Bsmt Full Bath': 0.6931471805599453,
        'Bsmt Half Bath': 0.0,
        'Full Bath': 1.0986122886681098,
        'Half Bath': 0.0,
        'Bedroom AbvGr': 1.6094379124341005,
        'Kitchen AbvGr': 0.6931471805599453,
        'TotRms AbvGrd': 2.079441541679836,
        'Fireplaces': 0.6931471805599453,
        'Garage Yr Blt': 7.576097340623111,
        'Garage Cars': 1.0986122886681098,
        'Garage Area': 6.3578422665081,
        'Wood Deck SF': 0.0,
        'Open Porch SF': 3.4011973816621555,
        'Enclosed Porch': 3.688879454113936,
        '3Ssn Porch': 0.0,
        'Screen Porch': 0.0,
        'Pool Area': 0.0,
        'Misc Val': 0.0,
        'Mo Sold': 1.9459101490553128,
        'Yr Sold': 7.60489448081162,
        'MS Zoning_RL': 1.0,
        'Street_Pave': 1.0,
        'Alley_missing': 1.0,
        'Lot Shape_Reg': 1.0,
        'Land Contour_Lvl': 1.0,
        'Lot Config_Inside': 0.0,
        'Neighborhood_Edwards': 1.0,
        'Condition 1_Norm': 1.0,
        'Condition 2_Norm': 1.0,
        'House Style_1Story': 1.0,
        'Roof Style_Gable': 1.0,
        'Exterior 1st_WdShing': 1.0,
        'Exterior 2nd_Wd Shng': 1.0,
        'Mas Vnr Type_missing': 1.0,
        'Exter Qual_TA': 1.0,
        'Exter Cond_TA': 1.0,
        'Foundation_CBlock': 1.0,
        'Bsmt Qual_TA': 1.0,
        'Bsmt Cond_Gd': 1.0,
        'BsmtFin Type 2_Unf': 1.0,
        'Heating_GasA': 1.0,
        'Heating QC_TA': 1.0,
        'Central Air_Y': 1.0,
        'Electrical_SBrkr': 1.0,
        'Kitchen Qual_Gd': 1.0,
        'Functional_Typ': 1.0,
        'Fireplace Qu_Gd': 1.0,
        'Garage Type_Attchd': 1.0,
        'Garage Finish_Unf': 1.0,
        'Garage Qual_TA': 1.0,
        'Garage Cond_TA': 1.0,
        'Paved Drive_Y': 1.0,
        'Fence_MnPrv': 1.0,
        'Misc Feature_missing': 1.0,
        'Sale Type_WD ': 1.0,
        'Sale Condition_Normal': 1.0
    }

def predict_price(input_data):
    """Predict house price based on input data"""
    try:
        model = load_model()
        if model is None:
            return None
        
        # Preprocess input data
        processed_data = preprocess_input(input_data)
        if processed_data is None:
            return None
        
        # Make prediction
        prediction = model.predict(processed_data)[0]
        
        # Convert back from log scale
        predicted_price = np.expm1(prediction)
        
        return round(predicted_price, 2)
    except Exception as e:
        print(f"Error making prediction: {e}")
        return None

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Get input data from command line arguments
        input_data = json.loads(sys.argv[1])
        result = predict_price(input_data)
        print(json.dumps({"predicted_price": result}))
    else:
        # Return sample data for testing
        sample_data = get_sample_data()
        print(json.dumps({"sample_data": sample_data})) 