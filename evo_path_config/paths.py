import os 

def get_path_config(evodata=[]):
    #  data loader for detection 
    M3FD_val_outfoler = "./datasets/M3FD/val/images"
    MSOD_val_outfoler = "./datasets/MSOD/val/images"
    LLVIP_val_outfoler = "./datasets/LLVIP/images/val"

    M3FD_val_irfoler = "./datasets/M3FD/val/irimages"
    MSOD_val_irfoler = "./datasets/MSOD/val/irimages"
    LLVIP_val_irfoler = "./datasets/LLVIP/irimages/val"

    M3FD_val_vifoler = "./datasets/M3FD/val/viimages"
    MSOD_val_vifoler = "./datasets/MSOD/val/viimages"
    LLVIP_val_vifoler = "./datasets/LLVIP/viimages/val"

    M3FD_val_loader = (M3FD_val_irfoler, M3FD_val_vifoler, 2, (512, 384))
    MSOD_val_loader = (MSOD_val_irfoler, MSOD_val_vifoler, 2, (512, 384))
    LLVIP_val_loader = (LLVIP_val_irfoler, LLVIP_val_vifoler, 2, (512, 384))


    M3FD_train_outfoler = "./datasets/M3FD/train/images"
    MSOD_train_outfoler = "./datasets/MSOD/train/images"
    LLVIP_train_outfoler = "./datasets/LLVIP/images/train"

    M3FD_train_irfoler = "./datasets/M3FD/train/irimages"
    MSOD_train_irfoler = "./datasets/MSOD/train/irimages"
    LLVIP_train_irfoler = "./datasets/LLVIP/irimages/train"

    M3FD_train_vifoler = "./datasets/M3FD/train/viimages"
    MSOD_train_vifoler = "./datasets/MSOD/train/viimages"
    LLVIP_train_vifoler = "./datasets/LLVIP/viimages/train"

    M3FD_train_loader = (M3FD_train_irfoler, M3FD_train_vifoler, 2, (512, 384))
    MSOD_train_loader = (MSOD_train_irfoler, MSOD_train_vifoler, 2, (512, 384))
    LLVIP_train_loader = (LLVIP_train_irfoler, LLVIP_train_vifoler, 2, (512, 384))


    #  data loader for segmentation 
    FMB_val_outfoler = "./datasets/FMB/test/images"
    MFNet_val_outfoler = "./datasets/MFNet/test/images"
    WHU_val_outfoler = "./datasets/WHU/val/images"
    potsdam_val_outfoler = "./datasets/potsdam/images/val"
    
    FMB_val_irfoler = "./datasets/FMB/test/Infrared"
    MFNet_val_irfoler = "./datasets/MFNet/test/Infrared"
    WHU_val_irfoler = "./datasets/WHU/val/Infrared"
    potsdam_val_irfoler = "./datasets/potsdam/ir_dir/val"
    
    FMB_val_vifoler = "./datasets/FMB/test/Visible"
    MFNet_val_vifoler = "./datasets/MFNet/test/Visible"
    WHU_val_vifoler = "./datasets/WHU/val/Visible"
    potsdam_val_vifoler = "./datasets/potsdam/img_dir/val"
    
    FMB_val_loader = (FMB_val_irfoler, FMB_val_vifoler, 2, (800, 600))
    MFNet_val_loader = (MFNet_val_irfoler, MFNet_val_vifoler, 2, (640, 480))
    WHU_val_loader = (WHU_val_irfoler, WHU_val_vifoler, 2, (1389, 926))
    potsdam_val_loader = (potsdam_val_irfoler, potsdam_val_vifoler, 2, (512, 512))

    FMB_train_outfoler = "./datasets/FMB/train/images"
    MFNet_train_outfoler = "./datasets/MFNet/train/images"
    WHU_train_outfoler = "./datasets/WHU/train/images"
    potsdam_train_outfoler = "./datasets/potsdam/images/train"

    FMB_train_irfoler = "./datasets/FMB/train/Infrared"
    MFNet_train_irfoler = "./datasets/MFNet/train/Infrared"
    WHU_train_irfoler = "./datasets/WHU/train/Infrared"
    potsdam_train_irfoler = "./datasets/potsdam/ir_dir/train"

    FMB_train_vifoler = "./datasets/FMB/train/Visible"
    MFNet_train_vifoler = "./datasets/MFNet/train/Visible"
    WHU_train_vifoler = "./datasets/WHU/train/Visible"
    potsdam_train_vifoler = "./datasets/potsdam/img_dir/train"
    
    FMB_train_loader = (FMB_train_irfoler, FMB_train_vifoler, 2, (800, 600))
    MFNet_train_loader = (MFNet_train_irfoler, MFNet_train_vifoler, 2, (640, 480))
    WHU_train_loader = (WHU_train_irfoler, WHU_train_vifoler, 2, (1389, 926))
    potsdam_train_loader = (potsdam_train_irfoler, potsdam_train_vifoler, 2, (512, 512))


    #  data loader for segmentation 
    VT821_val_outfoler = "./datasets/VT821/val/images"
    VT1000_val_outfoler = "./datasets/VT1000/val/images"
    VT5000_val_outfoler = "./datasets/VT5000/val/images"
    
    VT821_val_irfoler = "./datasets/VT821/val/T"
    VT1000_val_irfoler = "./datasets/VT1000/val/T"
    VT5000_val_irfoler = "./datasets/VT5000/val/T"
    
    VT821_val_vifoler = "./datasets/VT821/val/RGB"
    VT1000_val_vifoler = "./datasets/VT1000/val/RGB"
    VT5000_val_vifoler = "./datasets/VT5000/val/RGB"
    
    VT821_val_loader = (VT821_val_irfoler, VT821_val_vifoler, 2, (640, 480))
    VT1000_val_loader = (VT1000_val_irfoler, VT1000_val_vifoler, 2, (640, 480))
    VT5000_val_loader = (VT5000_val_irfoler, VT5000_val_vifoler, 2, (640, 480))

    VT821_train_outfoler = "./datasets/VT821/train/images"
    VT1000_train_outfoler = "./datasets/VT1000/train/images"
    VT5000_train_outfoler = "./datasets/VT5000/train/images"

    VT821_train_irfoler = "./datasets/VT821/train/T"
    VT1000_train_irfoler = "./datasets/VT1000/train/T"
    VT5000_train_irfoler = "./datasets/VT5000/train/T"

    VT821_train_vifoler = "./datasets/VT821/train/RGB"
    VT1000_train_vifoler = "./datasets/VT1000/train/RGB"
    VT5000_train_vifoler = "./datasets/VT5000/train/RGB"
    
    VT821_train_loader = (VT821_train_irfoler, VT821_train_vifoler, 2, (640, 480))
    VT1000_train_loader = (VT1000_train_irfoler, VT1000_train_vifoler, 2, (640, 480))
    VT5000_train_loader = (VT5000_train_irfoler, VT5000_train_vifoler, 2, (640, 480))
    
    (train_folder_list, train_irfolder_list, train_vifolder_list, train_loader_list, project_root_list,
    val_folder_list, val_irfolder_list, val_vifolder_list, val_loader_list) = [], [], [], [], [], [], [], [], []


    # best model ckpt path .    
    project_root = "./EvoFuseRuns/"
    
    
    M3FD_Model_Path = os.path.join(project_root, "M3FD/")
    MSOD_Model_Path = os.path.join(project_root, "MSOD/")
    LLVIP_Model_Path = os.path.join(project_root, "LLVIP/")
    FMB_Model_Path = os.path.join(project_root, "FMB/")
    MFNet_Model_Path = os.path.join(project_root, "MFNet/")
    WHU_Model_Path = os.path.join(project_root, "WHU/")
    potsdam_Model_Path = os.path.join(project_root, "potsdam/")
    VT821_Model_Path = os.path.join(project_root, "VT821/")
    VT1000_Model_Path = os.path.join(project_root, "VT1000/")
    VT5000_Model_Path = os.path.join(project_root, "VT5000/")
    
    
    for i in range(len(evodata)):
        if evodata[i] == "M3FD":
            train_folder_list.append(M3FD_train_outfoler)
            train_irfolder_list.append(M3FD_train_irfoler)
            train_vifolder_list.append(M3FD_train_vifoler)
            train_loader_list.append(M3FD_train_loader)
            val_folder_list.append(M3FD_val_outfoler)
            val_irfolder_list.append(M3FD_val_irfoler)
            val_vifolder_list.append(M3FD_val_vifoler)
            val_loader_list.append(M3FD_val_loader)
            project_root_list.append(M3FD_Model_Path)
        elif evodata[i] == "MSOD":
            train_folder_list.append(MSOD_train_outfoler)
            train_irfolder_list.append(MSOD_train_irfoler)
            train_vifolder_list.append(MSOD_train_vifoler)
            train_loader_list.append(MSOD_train_loader)
            val_folder_list.append(MSOD_val_outfoler)
            val_irfolder_list.append(MSOD_val_irfoler)
            val_vifolder_list.append(MSOD_val_vifoler)
            val_loader_list.append(MSOD_val_loader)
            project_root_list.append(MSOD_Model_Path)
        elif evodata[i] == "LLVIP":
            train_folder_list.append(LLVIP_train_outfoler)
            train_irfolder_list.append(LLVIP_train_irfoler)
            train_vifolder_list.append(LLVIP_train_vifoler)
            train_loader_list.append(LLVIP_train_loader)
            val_folder_list.append(LLVIP_val_outfoler)
            val_irfolder_list.append(LLVIP_val_irfoler)
            val_vifolder_list.append(LLVIP_val_vifoler)
            val_loader_list.append(LLVIP_val_loader)
            project_root_list.append(LLVIP_Model_Path)
        elif evodata[i] == "FMB":
            train_folder_list.append(FMB_train_outfoler)
            train_irfolder_list.append(FMB_train_irfoler)
            train_vifolder_list.append(FMB_train_vifoler)
            train_loader_list.append(FMB_train_loader)
            val_folder_list.append(FMB_val_outfoler)
            val_irfolder_list.append(FMB_val_irfoler)
            val_vifolder_list.append(FMB_val_vifoler)
            val_loader_list.append(FMB_val_loader)
            project_root_list.append(FMB_Model_Path)
        elif evodata[i] == "MFNet":
            train_folder_list.append(MFNet_train_outfoler)
            train_irfolder_list.append(MFNet_train_irfoler)
            train_vifolder_list.append(MFNet_train_vifoler)
            train_loader_list.append(MFNet_train_loader)
            val_folder_list.append(MFNet_val_outfoler)
            val_irfolder_list.append(MFNet_val_irfoler)
            val_vifolder_list.append(MFNet_val_vifoler)
            val_loader_list.append(MFNet_val_loader)
            project_root_list.append(MFNet_Model_Path)
        elif evodata[i] == "WHU":
            train_folder_list.append(WHU_train_outfoler)
            train_irfolder_list.append(WHU_train_irfoler)
            train_vifolder_list.append(WHU_train_vifoler)
            train_loader_list.append(WHU_train_loader)
            val_folder_list.append(WHU_val_outfoler)
            val_irfolder_list.append(WHU_val_irfoler)
            val_vifolder_list.append(WHU_val_vifoler)
            val_loader_list.append(WHU_val_loader)
            project_root_list.append(WHU_Model_Path)
        elif evodata[i] == "potsdam":
            train_folder_list.append(potsdam_train_outfoler)
            train_irfolder_list.append(potsdam_train_irfoler)
            train_vifolder_list.append(potsdam_train_vifoler)
            train_loader_list.append(potsdam_train_loader)
            val_folder_list.append(potsdam_val_outfoler)
            val_irfolder_list.append(potsdam_val_irfoler)
            val_vifolder_list.append(potsdam_val_vifoler)
            val_loader_list.append(potsdam_val_loader)
            project_root_list.append(potsdam_Model_Path)
        elif evodata[i] == "VT821":
            train_folder_list.append(VT821_train_outfoler)
            train_irfolder_list.append(VT821_train_irfoler)
            train_vifolder_list.append(VT821_train_vifoler)
            train_loader_list.append(VT821_train_loader)
            val_folder_list.append(VT821_val_outfoler)
            val_irfolder_list.append(VT821_val_irfoler)
            val_vifolder_list.append(VT821_val_vifoler)
            val_loader_list.append(VT821_val_loader)
            project_root_list.append(VT821_Model_Path)
        elif evodata[i] == "VT1000":
            train_folder_list.append(VT1000_train_outfoler)
            train_irfolder_list.append(VT1000_train_irfoler)
            train_vifolder_list.append(VT1000_train_vifoler)
            train_loader_list.append(VT1000_train_loader)
            val_folder_list.append(VT1000_val_outfoler)
            val_irfolder_list.append(VT1000_val_irfoler)
            val_vifolder_list.append(VT1000_val_vifoler)
            val_loader_list.append(VT1000_val_loader)
            project_root_list.append(VT1000_Model_Path)
        elif evodata[i] == "VT5000":
            train_folder_list.append(VT5000_train_outfoler)
            train_irfolder_list.append(VT5000_train_irfoler)
            train_vifolder_list.append(VT5000_train_vifoler)
            train_loader_list.append(VT5000_train_loader)
            val_folder_list.append(VT5000_val_outfoler)
            val_irfolder_list.append(VT5000_val_irfoler)
            val_vifolder_list.append(VT5000_val_vifoler)
            val_loader_list.append(VT5000_val_loader)
            project_root_list.append(VT5000_Model_Path)
        
        
        
    return (project_root, train_folder_list, train_irfolder_list, train_vifolder_list, train_loader_list,
            val_folder_list, val_irfolder_list, val_vifolder_list, val_loader_list, project_root_list)
    
    
    