
""" 
    I wanted to create a file that could be loaded into paraview and used to classify nodes as target nodes (T) or unconstrained nodes (U) or constrained nodes (C). 

    The problem with doing this was that paraview has its own version of python so we cant access all the usual modules that other scripts in this library can use.

    Therefore, this file had to be completely standalone, only using imports from the python standard library and paraviews custom library.

    To make sure the rest of my software could read the files, i included an interface class "MorphModelBase" that contains all the data 
    definitions that this script saves the data as. This can then be extended by a seperate class in another file so that it contains all the same data/datatypes, 
    but can include all processing functions and use modules from outside those available in paraview to process the nodes. 
"""

import os
import json

try:
    import paraview.simple as pvs
    import paraview.servermanager as sm
    from vtk.util.numpy_support import vtk_to_numpy
except ModuleNotFoundError:
    #print("Paraview library not loaded.")
    # Script is running outside of paraview. CreateMorph class will not work.
    pass

class MorphModelBase:
    """Base class definition of all parameters required to define a morph"""
    def __init__(self, f_name=None, con=False, pb=None, path=None, f=None, T=None, U=None, phi=0.0):
        self.f_name      = f_name
        self.path        = path
        self.constraint  = con
        self.phi_bounds  = pb if pb is not None else []
        self.phi         = phi
        self.bf_t        = "wendland_c_0"
        self.c_t         = "radius"
        self.c_mod_t     = 1.0
        self.bf_u        = "multiquadric"
        self.c_u         = "one"
        self.c_mod_u     = 1.0
        self.rigid_boundary_translation  = False
        self._f          = f
        self.const_prop  = 0

        self.t_sections  = T if T is not None else []
        self.t_node_ids  = {}
        self.t_face_ids  = {}
        self.t_surfaces  = []

        self.u_sections  = U if U is not None else []
        self.u_node_ids  = {}
        self.u_face_ids  = {}
        self.u_surfaces  = []

class CreateMorph:
    """
        A set of functions to setup a set of morphs to be applied to setup a case study
        users can define a morph, group surfaces into sections and save the selections to a file.
    """
    def __init__(self, name="default", path=None, mmbase=MorphModelBase):
        self.name     = name
        self.farfield = []
        self.sections = {}
        self.morphs   = []
        self.mmbase = mmbase
        self.current_morph = -1
        print("Setting up morph info.")
        self.name = "CaseFile"
        self.path = ""
        try:
            sources     = pvs.GetSources()
            for s in sources.keys():
                if hasattr(sources[s], "CaseFileName"):
                    fp = sources[s].CaseFileName
            path, name = os.path.split(fp)
            path += f"{os.sep}"
            name = name[:-5]
            if "ENSIGHT" in name:
                name = name[7:]
            self.name = name
            self.path = path
        except NameError:
            print("Setting up MorphInfo - paraview not loaded.")
        print(f"Morph name: {self.name}")
        print(f"Morph path: {self.path}")

    def read(self, path=None):
        """load morphs from a morph info file"""
        if path is None:
            path = self.path
        else:
            self.path = path
        data = None
        with open(f"{self.path}MorphInfo.json") as json_data:
            data = json.load(json_data)
            json_data.close()
        self.name     = data["name"]
        self.farfield = data["farfield"]
        self.const_prop = int(data.get("const_prop",1))
        for s in data["sections"]:
            k,v = list(s.items())[0]
            self.sections[k] = v
        for m in data["morphs"]:
            morph_model            = self.mmbase(m["function_name"])
            morph_model.path       = self.path
            morph_model.constraint = bool(m["constraint"])
            morph_model.bf_t       = m["bf_t"]
            morph_model.c_t        = m["c_t"]
            morph_model.c_mod_t    = float(m["c_mod_t"])
            morph_model.phi_bounds = m["phi_bounds"]
            morph_model.bf_u       = m["bf_u"]
            morph_model.c_u        = m["c_u"]
            morph_model.c_mod_u    = float(m["c_mod_u"])
            morph_model.t_sections = m["t_sections"]
            t_node_ids = {}
            for k,v in m["t_node_ids"].items():
                t_node_ids[int(k)] = v
            morph_model.t_node_ids  = t_node_ids
            t_face_ids = {}
            for k,v in m["t_face_ids"].items():
                t_face_ids[int(k)] = v
            morph_model.t_face_ids  = t_face_ids
            morph_model.t_surfaces  = m["t_surfaces"]
            morph_model.u_sections  = m["u_sections"]
            u_node_ids = {}
            for k,v in m["u_node_ids"].items():
                u_node_ids[int(k)] = v
            morph_model.u_node_ids  = u_node_ids
            u_face_ids = {}
            for k,v in m["u_face_ids"].items():
                u_face_ids[int(k)] = v
            morph_model.u_face_ids  = u_face_ids
            morph_model.u_surfaces  = m["u_surfaces"]
            self.morphs.append(morph_model)

    def write(self):
        """writes CreateMorphs to file using dunder str method"""
        with open(f"{self.path}MorphInfo.json","w") as f:
            f.write(str(self))

    def __str__(self):
        morph_info = {}
        morph_info["name"]     = self.name
        morph_info["farfield"] = self.farfield
        morph_info["const_prop"] = self.const_prop
        morph_info["sections"] = []
        for k,v in self.sections.items():
            morph_info["sections"].append({k:v})
        morph_info["morphs"] = []
        for v in self.morphs:
            morph = {}
            morph["function_name"] = v.f_name
            morph["constraint"] = v.constraint
            morph["rigid_boundary_translation"] = v.rigid_boundary_translation
            morph["phi_bounds"] = v.phi_bounds
            morph["bf_t"]       = v.bf_t
            morph["c_t"]        = v.c_t
            morph["c_mod_t"]    = v.c_mod_t
            morph["bf_u"]       = v.bf_u
            morph["c_u"]        = v.c_u
            morph["c_mod_u"]    = v.c_mod_u
            morph["t_sections"] = v.t_sections
            morph["t_node_ids"] = v.t_node_ids
            morph["t_face_ids"] = v.t_face_ids
            morph["t_surfaces"] = v.t_surfaces
            morph["u_sections"] = v.u_sections
            morph["u_node_ids"] = v.u_node_ids
            morph["u_face_ids"] = v.u_face_ids
            morph["u_surfaces"] = v.u_surfaces
            morph_info["morphs"].append(morph)
        return str(json.dumps(morph_info, indent=4))

    ## Morph management
    def add_new_morph(self, name=None):
        """create a new morph and make it the current selection"""
        if name is None:
            name = input("Please specify function:\n")
        self.current_morph = len(self.morphs)
        self.morphs.append(self.mmbase(name))

    def view_morphs(self):
        """
            print a list of all current morphs by name. current morph
            current morph is highlighted with a star
            returns a list dictionary of morphs (k: id, v:morph)
        """
        morphs = {}
        print("Current morphs")
        for i,morph in enumerate(self.morphs):
            morph_displayname = f"{i}) {morph.f_name}"
            if i == self.current_morph:
                morph_displayname += "*"
            print(morph_displayname)
            morphs[i] = morph
        return morphs

    def select_current_morph(self):
        """change the morph definition to be edited."""
        self.view_morphs()
        new_morph = int(input())
        self.current_morph = new_morph
        return self.get_current_morph()

    def get_current_morph(self):
        """display and return details about the currently selected morph definition"""
        if (len(self.morphs) == 0) or (self.current_morph >= len(self.morphs)):
            return None
        current_morph = self.morphs[self.current_morph]
        print(current_morph)
        return current_morph

    ## Section management
    # Allows you to group up multiple surfaces into one section. 
    # eg, can define a wing
    def add_section(self, name=None):
        """saves the currently selected surfaces as a section"""
        if name is None:
            name = input("Select section name:")
        surfaces = self.get_blocks()
        if len(surfaces) > 0:
            self.sections[name] = surfaces

    def view_sections(self):
        """view all currently available sections by name"""
        sections = {}
        print("Defined sections")
        for i,s in enumerate(list(self.sections.keys())):
            print(f"{i}) {s}")
            sections[i] = s
        return sections

    def view_section(self):
        """view all ids for a particular section"""
        sections = self.view_sections()
        try:
            s_id = int(input("Select section to view\n"))
            if s_id not in sections:
                print("Please select index that corresponds with one of the ids shown in the list")
                return None
            section = sections[s_id]
            print(f"Section {section} = {self.sections[section]}")
            return section
        except ValueError:
            print("Please select index that corresponds with one of the ids shown in the list")
            return None

    def select_sections(self):
        """select a group of sections from the currently defined list"""
        print("Select sections from the list below. -1 to complete selection, -2 to clear")
        current_selection = []
        inp = -3
        while inp != -1:
            sections = self.view_sections()
            inp_raw = input(f"Current selection:{current_selection}\n")
            if inp_raw == "":
                continue
            inp = int(inp_raw)
            if inp == -2:
                current_selection = []
            elif inp >= 0:
                if sections[inp] not in current_selection:
                    current_selection.append(sections[inp])
        return current_selection

    ## Node selection
    # T
    def select_t_sections(self):
        """
            selects a group of sections for the translatable 
            nodes from the currently defined list
        """
        self.morphs[self.current_morph].t_sections = self.select_sections()
        return self.get_current_morph()

    def select_t_nodes(self):
        """
            select a set of currently selected nodes to be the target nodes for 
            the currently selected morph
        """
        self.morphs[self.current_morph].t_node_ids = self.get_ids()
        return self.get_current_morph()

    def select_t_faces(self):
        """
            select a set of currently selected faces to be the target nodes for 
            the currently selected morph
        """
        self.morphs[self.current_morph].t_face_ids = self.get_ids()
        return self.get_current_morph()

    def select_t_surfaces(self):
        """
            select a set of currently selected surfaces (called blocks in paraview) to 
            be the target nodes for the currently selected morph
        """
        self.morphs[self.current_morph].t_surfaces = self.get_blocks()
        return self.get_current_morph()

    # U
    def select_u_sections(self):
        """
            selects a group of sections for the unconstrained 
            nodes from the currently defined list
        """
        self.morphs[self.current_morph].u_sections = self.select_sections()
        return self.get_current_morph()

    def select_u_nodes(self):
        """
            select a set of currently selected nodes to be the unconstrained nodes for 
            the currently selected morph
        """
        self.morphs[self.current_morph].u_node_ids = self.get_ids()
        return self.get_current_morph()

    def select_u_faces(self):
        """
            select a set of currently selected faces to be the unconstrained nodes for 
            the currently selected morph
        """
        self.morphs[self.current_morph].u_face_ids = self.get_ids()
        return self.get_current_morph()

    def select_u_surfaces(self):
        """
            select a set of currently selected surfaces (called blocks in paraview) to 
            be the unconstrained nodes for the currently selected morph
        """
        self.morphs[self.current_morph].u_surfaces = self.get_blocks()
        return self.get_current_morph()

    ## Utility - Paraview 5.11.1
    def get_blocks(self):
        """returns the surface ids that are currently selected."""
        
        # get currently loaded geometry
        case        = pvs.GetActiveSource()
        # create a temporary new part with all currently selected ids
        extract     = pvs.ExtractSelection(registrationName='temp', Input=case)
        # this is a paraview thing. aparently servermanager is how you access the
        # objects that allow you to access the coordinates
        obj         = sm.Fetch(pvs.GetActiveSource())
        num_blocks  = obj.GetNumberOfBlocks() # number of surface ids
        surface_ids = range(num_blocks)
        selected_surfaces = []
        for i in surface_ids:
            block = obj.GetBlock(i)
            # if the part has no selected nodes in it then GetBlock returns None.
            if block is None:
                continue
            selected_surfaces.append(i+1)
        pvs.Delete(extract)       # delete the temporary part
        del extract               # tidy up the local object we created for it.
        pvs.SetActiveSource(case) # reset the viewer to its original selection
        pvs.ClearSelection()      # unselects all nodes/faces/surfaces.
        return selected_surfaces

    def get_ids(self):
        """
            get a dictionary of the local ids (relative to its parent surface)
            of individually selected node ids
        """
        s_dict         = {}
        case           = pvs.GetActiveSource() # get currently loaded geometry
        # create a temporary new part with all currently selected ids
        extract        = pvs.ExtractSelection(registrationName='temp', Input=case)
        # this is a paraview thing. idk wtf a servermanager is but apparently
        # its how you access the raw coordinates...
        obj            = sm.Fetch(pvs.GetActiveSource())
        num_blocks     = obj.GetNumberOfBlocks() # number of surface ids
        surface_ids    = range(num_blocks)
        surface_points = {}
        for i in surface_ids:
            block = obj.GetBlock(i)
            # if the part has no selected nodes in it then GetBlock returns None
            if block is None: 
                continue
            # get the coordiantes of all selected poitns in the temporary extracted surface
            numpy_array = vtk_to_numpy(block.GetPoints().GetData())
            surface_points[i] = numpy_array
        pvs.Delete(extract) # clear the temporary surface 
        del extract
        pvs.SetActiveSource(case)
        # return if no points selected
        if len(surface_points) == 0:
            print("No points selected")
            return s_dict

        for surface_id, points in surface_points.items():
            # get a list of all points for the surface and convert them to string format for searchability
            sp_all = [str(a) for a in vtk_to_numpy(sm.Fetch(case).GetBlock(surface_id).GetPoints().GetData())]
            point_ids = []
            for p in points:
                point_ids.append(sp_all.index(str(p))) # loop through and get local indexes
            s_dict[surface_id+1] = point_ids # increment to match up with flites 1 indexed surface id's

        pvs.ClearSelection() # clear and return
        return s_dict
