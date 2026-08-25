import os 

def get_path_config(evodata=[]):
    #  data loader for fusion 
    M3FD_val_outfoler = "./FusionResults/M3FD/EvoFuse"
    TNO_val_outfoler = "./FusionResults/TNO/EvoFuse"
    RoadScene_val_outfoler = "./FusionResults/RoadScene/EvoFuse"
    FMB_val_outfoler = "./FusionResults/FMB/EvoFuse"

    M3FD_val_irfoler = "./FusionDatasets/M3FD/ir"
    TNO_val_irfoler = "./FusionDatasets/TNO/ir"
    RoadScene_val_irfoler = "./FusionDatasets/RoadScene/ir"
    FMB_val_irfoler = "./FusionDatasets/FMB/ir"

    M3FD_val_vifoler = "./FusionDatasets/M3FD/vi"
    TNO_val_vifoler = "./FusionDatasets/TNO/vi"
    RoadScene_val_vifoler = "./FusionDatasets/RoadScene/vi"
    FMB_val_vifoler = "./FusionDatasets/FMB/vi"

    M3FD_val_loader = (M3FD_val_irfoler, M3FD_val_vifoler, 2, (512, 384))
    TNO_val_loader = (TNO_val_irfoler, TNO_val_vifoler, 2, (512, 384))
    RoadScene_val_loader = (RoadScene_val_irfoler, RoadScene_val_vifoler, 2, (512, 384))
    FMB_val_loader = (FMB_val_irfoler, FMB_val_vifoler, 2, (800, 600))



    
    
    (train_folder_list, train_irfolder_list, train_vifolder_list, train_loader_list, project_root_list,
    val_folder_list, val_irfolder_list, val_vifolder_list, val_loader_list) = [], [], [], [], [], [], [], [], []

    
    project_root = "./FusionResults/"
    
    M3FD_Model_Path = os.path.join(project_root, "M3FD/")
    TNO_Model_Path = os.path.join(project_root, "TNO/")
    RoadScene_Model_Path = os.path.join(project_root, "RoadScene/")
    FMB_Model_Path = os.path.join(project_root, "FMB/")
    
    
    for i in range(len(evodata)):
        if evodata[i] == "M3FD":
            val_folder_list.append(M3FD_val_outfoler)
            val_irfolder_list.append(M3FD_val_irfoler)
            val_vifolder_list.append(M3FD_val_vifoler)
            val_loader_list.append(M3FD_val_loader)
            project_root_list.append(M3FD_Model_Path)
        elif evodata[i] == "TNO":
            val_folder_list.append(TNO_val_outfoler)
            val_irfolder_list.append(TNO_val_irfoler)
            val_vifolder_list.append(TNO_val_vifoler)
            val_loader_list.append(TNO_val_loader)
            project_root_list.append(TNO_Model_Path)
        elif evodata[i] == "RoadScene":
            val_folder_list.append(RoadScene_val_outfoler)
            val_irfolder_list.append(RoadScene_val_irfoler)
            val_vifolder_list.append(RoadScene_val_vifoler)
            val_loader_list.append(RoadScene_val_loader)
            project_root_list.append(RoadScene_Model_Path)
        elif evodata[i] == "FMB":
            val_folder_list.append(FMB_val_outfoler)
            val_irfolder_list.append(FMB_val_irfoler)
            val_vifolder_list.append(FMB_val_vifoler)
            val_loader_list.append(FMB_val_loader)
            project_root_list.append(FMB_Model_Path)
        
        
        
    return (project_root, train_folder_list, train_irfolder_list, train_vifolder_list, train_loader_list,
            val_folder_list, val_irfolder_list, val_vifolder_list, val_loader_list, project_root_list)
    
    
    