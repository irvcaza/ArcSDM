﻿import sys
import arcpy

import arcsdm;

from arcsdm import *
from arcsdm.sitereduction import ReduceSites
from arcsdm.calculateweights import Calculate
from arcsdm.categoricalmembership import Calculate
from arcsdm.logisticregression import Execute
from arcsdm.calculateresponse import Execute
from arcsdm.common import reload_module, execute_tool
from arcsdm.symbolize import execute


import importlib
from imp import reload;


class Toolbox(object):
    def __init__(self):
        """Define the toolbox (the name of the toolbox is the name of the
        .pyt file)."""
        
        self.label = "ArcSDM toolbox"
        self.alias = "ArcSDM" 

        # List of tool classes associated with this toolbox
        self.tools = [CalculateWeightsTool,SiteReductionTool,CategoricalMembershipToool,CategoricalAndReclassTool, TOCFuzzificationTool, CalculateResponse, LogisticRegressionTool, Symbolize]

class Symbolize(object):
    def __init__(self):
        """Define the tool (tool name is the name of the class)."""
        self.label = "Symbolize raster with priorprobability (classified values)"
        self.description = "TODO: Describe this"
        self.canRunInBackground = False
        self.category = "Utilities"

    def getParameterInfo(self):
        """Define parameter definitions"""
        param0 = arcpy.Parameter(
        displayName="Raster Layer to symbolize",
        name="evidence_raster_layer",
        datatype="GPRasterLayer",
        parameterType="Required",
        direction="Input")

        
        param2 = arcpy.Parameter(
        displayName="Training sites (for prior prob)",
        name="training_sites",
        #datatype="DEFeatureClass",
        datatype="GPFeatureLayer",
        parameterType="Required",
        direction="Input")
        
        param5 = arcpy.Parameter(
        displayName="Unit area (km2)",
        name="Unit_Area__sq_km_",
        datatype="GPDouble",
        parameterType="Required",
        direction="Input")
        param5.value = "1";
        
                                  
        params = [param0, param2, param5]
        return params

    def isLicensed(self):
        """Set whether tool is licensed to execute."""
        return True

    def updateParameters(self, parameters):
        """Modify the values and properties of parameters before internal
        validation is performed.  This method is called whenever a parameter
        has been changed."""
        return

    def updateMessages(self, parameters):
        """Modify the messages created by internal validation for each tool
        parameter.  This method is called after internal validation."""
     
        return

    def execute(self, parameters, messages):
        """The source code of the tool."""
        #3.4
        try:
            importlib.reload (arcsdm.symbolize)
        except :
            reload(arcsdm.symbolize)
        arcsdm.symbolize.execute(self, parameters, messages)
        return
        
                       
        
class CalculateResponse(object):
    def __init__(self):
        """Define the tool (tool name is the name of the class)."""
        self.label = "Calculate response"
        self.description = "TODO: Copy this from old toolbox"
        self.canRunInBackground = False
        self.category = "Weights of Evidence"

    def getParameterInfo(self):
        """Define parameter definitions"""
        # TODO: Multiple rasters?
        param0 = arcpy.Parameter(
        displayName="Input Raster Layer(s)",
        name="Input_evidence_raster_layers",
        datatype="GPValueTable",
        parameterType="Required",
        direction="Input")
        param0.columns = [['GPRasterLayer', 'Evidence raster']]
        
        param1 = arcpy.Parameter(
        displayName="Evidence type",
        name="Evidence_Type",
        datatype="GPValueTable",
        parameterType="Required",
        direction="Input")
        param1.columns = [['GPString', 'Evidence type']]
        param1.parameterDependencies = ["0"];
        
        #param1.filter.type = "ValueList";
        #param1.filter.list = ["o", "c"];
        #param1.value = "o";
        
        paramInputWeights = arcpy.Parameter(
        displayName="Input weights tables",
        name="input_weights_tables",
        datatype="GPValueTable",
        parameterType="Required",
        direction="Input")
        paramInputWeights.columns = [['DETable', 'Weights table']]
        
        
        param2 = arcpy.Parameter(
        displayName="Training sites",
        name="training_sites",
        #datatype="DEFeatureClass",
        datatype="GPFeatureLayer",
        parameterType="Required",
        direction="Input")
        
        
        paramIgnoreMissing = arcpy.Parameter(
        displayName="Ignore missing data",
        name="Ignore missing data",
        datatype="Boolean",
        parameterType="Optional",
        direction="Output")
        #paramIgnoreMissing.value= false;
        
        
        param3 = arcpy.Parameter(
        displayName="Missing data value",
        name="Missing_Data_Value",
        datatype="GPLong",
        
        #parameterType="Required",
        direction="Output")
        param3.value= -99;
        

        param4 = arcpy.Parameter(
        displayName="Unit area (km^2)",
        name="Unit_Area_sq_km",
        datatype="GPDouble",
        parameterType="Required",
        direction="Input")
        param4.value = "1";
        
        
        
        
        param_pprb = arcpy.Parameter(
        displayName="Output post probablity raster",
        name="Output_Post_Probability_raster",
        datatype="DERasterDataset",
        parameterType="Required",
        direction="Output")
        param_pprb.value = "%Workspace%\W_pprb"
        
        
        param_std = arcpy.Parameter(
        displayName="Output standard deviation raster",
        name="Output_Standard_Deviation_raster",
        datatype="DERasterDataset",
        parameterType="Required",
        direction="Output")
        param_std.value = "%Workspace%\W_std"
        
        
        param_md_varianceraster = arcpy.Parameter(
        displayName="Output MD variance raster",
        name="output_md_variance_raster",
        datatype="DEDbaseTable",
        parameterType="Required",
        direction="Output")
        param_md_varianceraster.value = "%Workspace%\W_MDvar"
                
        param_totstddev = arcpy.Parameter(
        displayName="Output Total Std Deviation Raster",
        name="output_total_std_dev_raster",
        datatype="DEDbaseTable",
        parameterType="Required",
        direction="Output")
        param_totstddev.value = "%Workspace%\W_Tstd"
                                            
        
        param_Confraster = arcpy.Parameter(
        displayName="Output confidence raster",
        name="Output_Confidence_raster",
        datatype="DERasterDataset",
        parameterType="Required",
        direction="Output")
        param_Confraster.value = "%Workspace%\W_conf"
        
        
                                  
        params = [param0, paramInputWeights, param2, paramIgnoreMissing, param3, param4,  param_pprb, param_std, param_md_varianceraster, param_totstddev,  param_Confraster]
        return params

    def isLicensed(self):
        """Set whether tool is licensed to execute."""
        return True

    def updateParameters(self, parameters):
        """Modify the values and properties of parameters before internal
        validation is performed.  This method is called whenever a parameter
        has been changed."""
        return

    def updateMessages(self, parameters):
        """Modify the messages created by internal validation for each tool
        parameter.  This method is called after internal validation."""
     
        return

    def execute(self, parameters, messages):
        """The source code of the tool."""
        #3.4
        try:
            importlib.reload (arcsdm.calculateresponse)
        except :
            reload(arcsdm.calculateresponse);
        # To list what functions does module contain
        #messages.addWarningMessage(dir(arcsdm.SiteReduction));
        #arcsdm.CalculateWeights.Calculate(self, parameters, messages);
        #messages.AddMessage("Waiting for debugger")
        #wait_for_debugger(15);
        #No do yet
        arcsdm.calculateresponse.Execute(self, parameters, messages)
        return
        
                

        
        

class CalculateWeightsTool(object):
    def __init__(self):
        """Define the tool (tool name is the name of the class)."""
        self.label = "Calculate Weights"
        self.description = "Calculate weight rasters from the inputs"
        self.canRunInBackground = False
        self.category = "Weights of Evidence"

    def getParameterInfo(self):
        """Define parameter definitions"""
        param0 = arcpy.Parameter(
        displayName="Evidence Raster Layer",
        name="evidence_raster_layer",
        datatype="GPRasterLayer",
        parameterType="Required",
        direction="Input")

        param1 = arcpy.Parameter(
        displayName="Evidence raster codefield",
        name="Evidence_Raster_Code_Field",
        datatype="Field",
        parameterType="Optional",
        direction="Input")

        paramTrainingPoints = arcpy.Parameter(
        displayName="Training points",
        name="Training_points",
        datatype="GPFeatureLayer",
        parameterType="Required",
        direction="Input")
        
        param2 = arcpy.Parameter(
        displayName="Type",
        name="Type",
        datatype="GPString",
        parameterType="Required",
        direction="Input")
        param2.filter.type = "ValueList";
        param2.filter.list = ["Descending", "Ascending", "Categorical", "Unique"];
        param2.value = "Descending";
        
        param3 = arcpy.Parameter(
        displayName="Output weights table",
        name="output_weights_table",
        datatype="DETable",
        parameterType="Required",
        direction="Output")

        param4 = arcpy.Parameter(
        displayName="Confidence Level of Studentized Contrast",
        name="Confidence_Level_of_Studentized_Contrast",
        datatype="GPDouble",
        parameterType="Required",
        direction="Input")
        param4.value = "2";
                           
        param5 = arcpy.Parameter(
        displayName="Unit area (km2)",
        name="Unit_Area__sq_km_",
        datatype="GPDouble",
        parameterType="Required",
        direction="Input")
        param5.value = "1";
        
        param6 = arcpy.Parameter(
        displayName="Missing data value",
        name="Missing_Data_Value",
        datatype="GPLong",
        parameterType="Required",
        direction="Input")
        param6.value = "-99";

                           
                                  
        params = [param0, param1, paramTrainingPoints, param2, param3, param4, param5, param6]
        return params

    def isLicensed(self):
        """Set whether tool is licensed to execute."""
        return True

    def updateParameters(self, parameters):
        """Modify the values and properties of parameters before internal
        validation is performed.  This method is called whenever a parameter
        has been changed."""
        if parameters[0].value and parameters[3].value:
            if parameters[0].altered or paramaters[3].altered:
                layer = parameters[0].valueAsText;
                desc = arcpy.Describe(layer)
                name = desc.file;
                type = parameters[3].valueAsText;
                
                #Update name accordingly
                parameters[4].value = "%WORKSPACE%\\" + name + "_W" + type[:1]; #Output is _W + first letter of type
        
        
        return

    def updateMessages(self, parameters):
        """Modify the messages created by internal validation for each tool
        parameter.  This method is called after internal validation."""
     
        return

    def execute(self, parameters, messages):
        """The source code of the tool."""
        #3.4
        try:
            importlib.reload (arcsdm.calculateweights)
        except :
            reload(arcsdm.calculateweights);
        # To list what functions does module contain
        #messages.addWarningMessage(dir(arcsdm.SiteReduction));
        #arcsdm.CalculateWeights.Calculate(self, parameters, messages);
        #messages.AddMessage("Waiting for debugger")
        #wait_for_debugger(15);
        calculateweights.Calculate(self, parameters, messages)
        
        return
        
        

        
        
        
class SiteReductionTool(object):
    def __init__(self):
        """Define the tool (tool name is the name of the class)."""
        self.label = "Training sites reduction"
        self.description = "Selects subset of the training points"
        self.canRunInBackground = False
        self.category = "Weights of Evidence"

    def getParameterInfo(self):
        """Define parameter definitions"""
        param0 = arcpy.Parameter(
        displayName="Training sites layer",
        name="Training_Sites_layer",
        datatype="GPFeatureLayer",
        parameterType="Required",
        direction="Input")

        param1 = arcpy.Parameter(
        displayName="Thinning selection",
        name="Thinning_Selection",
        datatype="Boolean",
        parameterType="Optional",
        direction="Input")

        param2 = arcpy.Parameter(
        displayName="Unit area (sq km)",
        name="Unit_Area__sq_km_",
        datatype="GPLong",
        parameterType="Optional",
        direction="Input")
        
# Tämä vois olla hyvinkin valintalaatikko?
        param3 = arcpy.Parameter(
        displayName="Random selection",
        name="Random_selection",
        datatype="Boolean",
        parameterType="Optional",
        direction="Input")

        param4 = arcpy.Parameter(
        displayName="Random percentage selection",
        name="Random_percentage_selection",
        datatype="GPLong",
        parameterType="Optional",
        direction="Input")
                                            
        params = [param0, param1, param2, param3, param4]
        return params

    def isLicensed(self):
        """Set whether tool is licensed to execute."""
        return True

    def updateParameters(self, parameters):
        """Modify the values and properties of parameters before internal
        validation is performed.  This method is called whenever a parameter
        has been changed."""
        if parameters[1].value == True:
            parameters[2].enabled = True;
        else:
            parameters[2].enabled = False;            
            
        if parameters[3].value == True:
            parameters[4].enabled = True;
        
        else:
            parameters[4].enabled = False;  
        return

    def updateMessages(self, parameters):
        """Modify the messages created by internal validation for each tool
        parameter.  This method is called after internal validation."""
        l = 0;
        
        if parameters[1].value == True:
            #parameters[4].setErrorMessage("Random percentage value required!")
            l = l + 1;
            if parameters[2].value == '' or  parameters[2].value is None:
                parameters[2].setErrorMessage("Thinning value required!")
                
        if parameters[3].value == True:
            l = l + 1;            
            #parameters[4].setErrorMessage("Random percentage value required!")
            if parameters[4].value == '' or  parameters[4].value is None:
                parameters[4].setErrorMessage("Random percentage value required!")
            elif parameters[4].value > 100 or parameters[4].value < 1:
                parameters[4].setErrorMessage("Value has to between 1-100 %!")
            
        if (l < 1):
            parameters[1].setErrorMessage("You have to select either one!")
            parameters[3].setErrorMessage("You have to select either one!")
        
        return

    def execute(self, parameters, messages):
        """The source code of the tool."""
        try:
            importlib.reload (arcsdm.sitereduction)
        except :
            reload(sitereduction);
        sitereduction.ReduceSites(self, parameters, messages)
        return
        
class CategoricalMembershipToool(object):
    def __init__(self):
        """Define the tool (tool name is the name of the class)."""
        self.label = "Categorical Membership"
        self.description = "Create fuzzy memberships for categorical data by first reclassification to integers and then division by an appropriate value"
        self.canRunInBackground = False
        self.category = "Fuzzy Logic\\Fuzzy Membership"

    def getParameterInfo(self):
        """Define parameter definitions"""
        param0 = arcpy.Parameter(
        displayName="Categorical evidence raster",
        name="categorical_evidence",
        datatype="GPRasterLayer",
        parameterType="Required",
        direction="Input")

        param1 = arcpy.Parameter(
        displayName="Reclassification",
        name="reclassification",
        datatype="GPTableView",
        parameterType="Required",
        direction="Input")

        param2 = arcpy.Parameter(
        displayName="Rescale Constant",
        name="rescale_constant",
        datatype="GPLong",
        parameterType="Required",
        direction="Input")

        param3 = arcpy.Parameter(
        displayName="FMCat",
        name="fmcat",
        datatype="DERasterDataset",
        parameterType="Required",
        direction="Output")
        
        params = [param0, param1, param2, param3]
        return params

    def isLicensed(self):
        """Set whether tool is licensed to execute."""
        return True

    def updateParameters(self, parameters):
        """Modify the values and properties of parameters before internal
        validation is performed.  This method is called whenever a parameter
        has been changed."""
        return

    def updateMessages(self, parameters):
        """Modify the messages created by internal validation for each tool
        parameter.  This method is called after internal validation."""
        return

    def execute(self, parameters, messages):
        """The source code of the tool."""
        try:
            importlib.reload (arcsdm.categoricalmembership)
        except:
            reload(arcsdm.categoricalmembership)
        categoricalmembership.Calculate(self, parameters, messages)
        return

class CategoricalAndReclassTool(object):
    def __init__(self):
        """Define the tool (tool name is the name of the class)."""
        self.label = "Categorical & Reclass"
        self.description = "Create fuzzy memberships for categorical data by first reclassification to integers and then division by an appropriate value."
        self.canRunInBackground = False
        self.category = "Fuzzy Logic\\Fuzzy Membership"

    def getParameterInfo(self):
        """Define parameter definitions"""
        param0 = arcpy.Parameter(
        displayName="Categorical evidence raster",
        name="categorical_evidence",
        datatype="GPRasterLayer",
        parameterType="Required",
        direction="Input")
        
        param1 = arcpy.Parameter(
        displayName="Reclass field",
        name="reclass_field",
        datatype="Field",
        parameterType="Required",
        direction="Input")
        
        param2 = arcpy.Parameter(
        displayName="Reclassification",
        name="reclassification",
        datatype="remap",
        parameterType="Required",
        direction="Input")
        
        param3 = arcpy.Parameter(
        displayName="FM Categorical",
        name="fmcat",
        datatype="DERasterDataset",
        parameterType="Required",
        direction="Output")

        param4 = arcpy.Parameter(
        displayName="Divisor",
        name="divisor",
        datatype="GPLong",
        parameterType="Required",
        direction="Input")
        
        param1.value = "VALUE"
        param1.enabled = False
        param2.enabled = False
        param1.parameterDependencies = [param0.name]  
        param2.parameterDependencies = [param0.name,param1.name]

        params = [param0,param1,param2,param3,param4]
        return params

    def isLicensed(self):
        """Set whether tool is licensed to execute."""
        return True
    
    def updateParameters(self, parameters):
        """Modify the values and properties of parameters before internal
        validation is performed.  This method is called whenever a parameter
        has been changed."""
        if parameters[0].value:
            parameters[1].enabled = True
            parameters[2].enabled = True
        else:
            parameters[1].enabled = False
            parameters[2].enabled = False
        return

    def updateMessages(self, parameters):
        """Modify the messages created by internal validation for each tool
        parameter.  This method is called after internal validation."""
        return

    def execute(self, parameters, messages):
        """The source code of the tool."""
        try:
            importlib.reload (arcsdm.categoricalreclass)
        except:
            reload(arcsdm.categoricalreclass)
        categoricalreclass.Calculate(self, parameters, messages)
        return

class TOCFuzzificationTool(object):
    def __init__(self):
        """Define the tool (tool name is the name of the class)."""
        self.label = "TOC Fuzzification"
        self.description = "This fuzzification method utilized the symbolization of the input raster that has been applied in the map document table of contects (TOC). The symbolization in the TOC defines the number of classes and this tool rescales those classes (1...N) to the range [0,1] by (C - 1)/(N-1) where C is the class value and N is the number of classes."
        self.canRunInBackground = False
        self.category = "Fuzzy Logic\\Fuzzy Membership"

    def getParameterInfo(self):
        """Define parameter definitions"""
        param0 = arcpy.Parameter(
        displayName="Input Raster",
        name="input_raster",
        datatype="GPRasterLayer",
        parameterType="Required",
        direction="Input")
        
        param1 = arcpy.Parameter(
        displayName="Reclass Field",
        name="reclass_field",
        datatype="Field",
        parameterType="Required",
        direction="Input")

        param2 = arcpy.Parameter(
        displayName="Reclassification",
        name="reclassification",
        datatype="remap",
        parameterType="Required",
        direction="Output")

        param3 = arcpy.Parameter(
        displayName="Number of Classes",
        name="classes",
        datatype="GPLong",
        parameterType="Required",
        direction="Input")

        param4 = arcpy.Parameter(
        displayName="Fuzzy Membership Raster",
        name="fmtoc",
        datatype="DERasterDataset",
        parameterType="Required",
        direction="Output")
        
        param1.value = "VALUE"
        param1.parameterDependencies = [param0.name]  
        param2.parameterDependencies = [param0.name,param1.name]
        params = [param0,param1,param2,param3,param4]
        return params

    def isLicensed(self):
        """Set whether tool is licensed to execute."""
        return True
    
    def updateParameters(self, parameters):
        """Modify the values and properties of parameters before internal
        validation is performed.  This method is called whenever a parameter
        has been changed."""
        return

    def updateMessages(self, parameters):
        """Modify the messages created by internal validation for each tool
        parameter.  This method is called after internal validation."""
        return

    def execute(self, parameters, messages):
        """The source code of the tool."""
        try:
            importlib.reload (arcsdm.tocfuzzification)
        except:
            reload(arcsdm.tocfuzzification)
        tocfuzzification.Calculate(self, parameters, messages)
        return
        
class LogisticRegressionTool(object):
    def __init__(self):
        """Define the tool (tool name is the name of the class)."""
        self.label = "Logistic regression"
        self.description = "TODO: Copy this from old toolbox"
        self.canRunInBackground = False
        self.category = "Weights of Evidence"


    def getParameterInfo(self):
        """Define parameter definitions"""
        param0 = arcpy.Parameter(
        displayName="Input Raster Layer(s)",
        name="Input_evidence_raster_layers",
        datatype="GPValueTable",
        parameterType="Required",
        direction="Input")
        param0.columns = [['GPRasterLayer', 'Evidence raster']]
        
        param1 = arcpy.Parameter(
        displayName="Evidence type",
        name="Evidence_Type",
        datatype="GPValueTable",
        parameterType="Required",
        direction="Input")
        param1.columns = [['GPString', 'Evidence type']]
        param1.parameterDependencies = ["0"];
        
        #param1.filter.type = "ValueList";
        #param1.filter.list = ["o", "c"];
        #param1.value = "o";
        
        paramInputWeights = arcpy.Parameter(
        displayName="Input weights tables",
        name="input_weights_tables",
        datatype="GPValueTable",
        parameterType="Required",
        direction="Input")
        paramInputWeights.columns = [['DETable', 'Weights table']]

        param2 = arcpy.Parameter(
        displayName="Training sites",
        name="training_sites",
        #datatype="DEFeatureClass",
        datatype="GPFeatureLayer",
        parameterType="Required",
        direction="Input")
        
        param3 = arcpy.Parameter(
        displayName="Missing data value",
        name="Missing_Data_Value",
        datatype="GPLong",
        parameterType="Required",
        direction="Output")
        param3.value= -99;

        param4 = arcpy.Parameter(
        displayName="Unit area (km^2)",
        name="Unit_Area_sq_km",
        datatype="GPDouble",
        parameterType="Required",
        direction="Input")
        param4.value = "1";
        
        param5 = arcpy.Parameter(
        displayName="Output polynomial table",
        name="Output_Polynomial_Table",
        datatype="DEDbaseTable",
        parameterType="Required",
        direction="Output")
        param5.value = "%Workspace%\LR_logpol"
                
        param52 = arcpy.Parameter(
        displayName="Output coefficients table",
        name="Output_Coefficients_Table",
        datatype="DEDbaseTable",
        parameterType="Required",
        direction="Output")
        param52.value = "%Workspace%\LR_coeff"
        
        param6 = arcpy.Parameter(
        displayName="Output post probablity raster",
        name="Output_Post_Probability_raster",
        datatype="DERasterDataset",
        parameterType="Required",
        direction="Output")
        param6.value = "%Workspace%\LR_pprb"
        
        param62 = arcpy.Parameter(
        displayName="Output standard deviation raster",
        name="Output_Standard_Deviation_raster",
        datatype="DERasterDataset",
        parameterType="Required",
        direction="Output")
        param62.value = "%Workspace%\LR_std"
        
        param63 = arcpy.Parameter(
        displayName="Output confidence raster",
        name="Output_Confidence_raster",
        datatype="DERasterDataset",
        parameterType="Required",
        direction="Output")
        param63.value = "%Workspace%\LR_conf"
                                  
        params = [param0, param1, paramInputWeights, param2, param3, param4, param5, param52, param6, param62, param63]
        return params

    def isLicensed(self):
        """Set whether tool is licensed to execute."""
        return True

    def updateParameters(self, parameters):
        """Modify the values and properties of parameters before internal
        validation is performed.  This method is called whenever a parameter
        has been changed."""
        return

    def updateMessages(self, parameters):
        """Modify the messages created by internal validation for each tool
        parameter.  This method is called after internal validation."""
        return

    def execute(self, parameters, messages):
        """The source code of the tool."""
        reload_module(arcsdm.common, messages)
        reload_module(arcsdm.sdmvalues, messages)
        reload_module (arcsdm.workarounds_93, messages)
        reload_module (arcsdm.logisticregression, messages)
        execute_tool(arcsdm.logisticregression.Execute, self, parameters, messages)
        return