## Visual Genome Dataset

### Step 1
Download the annotations and images from the Visual Genome website http://visualgenome.org/api/v0/api_home.html and
download VG-SGG-with-attri.h5 file from https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch/blob/master/DATASET.md.
Put them under the datasets/vg directory.

	- datasets
	  - vg
	    - annottions
	      - image_data.json
	      - objects.json
	      - relationships.json
	    - images
	      - VG_100K
		- 2377357.jpg
		- 2377380.jpg
		- ...
	      - VG_100K_2
		- 2417985.jpg
		- 2417973.jpg
		- ...
      - VG-SGG-with-attri.h5

### Step 2 
Execute ```./preprocess.py``` to pre-process the dataset annotations, and the following two files will be saved to the datasets/vg/annottions directory.
We implement in our own code the same dataset pre-processing techniques in https://github.com/danfeiX/scene-graph-TF-release/tree/master/data_tools.
No GPU is required at this step.
#### You can SKIP this step by directly downloading and using the processed annotations provided in step 3.

	- datasets
	  - vg
	    - annottions
	      - instances_vg_train.json
	      - instances_vg_test.json
	      - ...
            - images
	      ...

	We provide the pre-processed dataset annotations here
	instances_vg_train: https://drive.google.com/file/d/1xEUk6jqZtKE0Myrs1I-lPoQZ3-2sTCjF/view?usp=sharing
	instances_vg_test.json: https://drive.google.com/file/d/1qkiXDw9sMsyCOMB-avBYnF8VGIqds0_k/view?usp=sharing


### Step 3
Execute ```./prepare_datasets.py``` to build the dataloader offline to speed up the later training process.
Pre-processed annotations in step 2 will be loaded to form an annotation for each image data, which is ready to be used in the scene graph training process.
One GPU is required at this step.
Each annotation for each image will be saved under the datasets/vg_scene_graph_annot directory as follows

	- datasets
	  - vg
	    ...
	  - vg_scene_graph_annot
	    - VG_100K
	      - 2377357_annotations.pkl
	      - 2377380_annotations.pkl
	      - ...
	    - VG_100K_2
	      - 2417985_annotations.pkl
	      - 2417973_annotations.pkl
	      - ...
	    ...

#### We provide all processed annotations here (it takes 2.09GB after unzipping)
https://drive.google.com/file/d/1hPLP-6Ub7s7zCthrfO-C2b_f_agXCiGv/view?usp=sharing


## OpenImage V6 Dataset

### Step 1
Download the processed OpenImage V6 dataset provided by [https://github.com/Scarecrow0/SGTR](https://github.com/Scarecrow0/SGTR/blob/main/DATASET.md) and put them under the datasets/ directory as datasets/open_image_v6. (it takes about 38GB)

### Step 2
We provide the depth maps for all images here. Download and put them under the datasets/open_image_v6 directory. (it takes about 640MB)
https://drive.google.com/file/d/1c5U-TG6hVPyjD6rdD6hgmp72HCq4mkXU/view?usp=sharing

## The final dataset directory should have the following structure
	- datasets
	  - vg
	    - annottions
	      - image_data.json
	      - objects.json
	      - relationships.json
	      - instances_vg_train.json
	      - instances_vg_test.json
	    - images
	      - VG_100K
	        - 2377357.jpg
	        - 2377380.jpg
	        - ...
	      - VG_100K_2
		- 2417985.jpg
		- 2417973.jpg
		- ...
       - VG-SGG-with-attri.h5
	  - vg_scene_graph_annot
	    - VG_100K
	      - 2377357_annotations.pkl
	      - 2377380_annotations.pkl
	      - ...
	    - VG_100K_2
	      - 2417985_annotations.pkl
	      - 2417973_annotations.pkl
	      - ...
	    
	  - open_image_v6
	    - annottions
	      - oiv6-adjust
	        - vrd-train-anno.json
		- vrd-test-anno.json
		- vrd-train-anno.json
		- vrd-val-anno.json
	    - image_depths
	      - 0a0b34cd17d2a797_depth.pt
	      - 00a0d634ad200ced_depth.pt
	      - ...
	    - images
	      - 0a0b34cd17d2a797.jpg
	      - 00a0d634ad200ced.jpg
	      - ...
