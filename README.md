## CShaper

Implementation for *Establishment of a morphological atlas of the Caenorhabditis elegans embryo using deep-learning-based 4D
 segmentation*, by Jianfeng Cao<sup>#</sup>, Guoye Guan<sup>#</sup>, Vincy Wing Sze Ho<sup>#</sup>, Ming-Kin Wong, Lu-Yan Chan, Chao Tang, Zhongying Zhao, & Hong Yan.
 
<sup>#</sup> equal contribution

### Usage
This implementation is based on Tensorflow and python3.6, trained with one GPU NVIDIA 2080Ti on Linux. Steps for training
and testing are listed as below.
* **Intsall dependency library**:
```buildoutcfg
    pip install requirements.txt
```
* **Train**: Download the data from this link (TBD) and put it into `./Data` folder, Set parameters
in `./ConfigMemb/train_edt_discrete.txt`, then run
    ```buildoutcfg
    python train.py ./ConfigMemb/train_edt_discrete.txt
    ```
* **Test**: Put the raw data (membrane and nucleus stack, and CD files from [AceTree](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC1501046/))
into `./Data/MembValidation/`. The pretrained model is available through the [Google Drive](https://drive.google.com/file/d/1ZwKKqAwVWr8YGGtdal-ZVxodyE7PUnb6/view?usp=sharing), which should be unzip to `./ModelCell/`. Example data is also available through previous data link. Set parameters in 
`./ConfigMemb/test_edt_discrete.txt` and run
    ```buildoutcfg
    python test_edt.py ./ConfigMemb/test_edt_discrete.txt
    ```
    Then binary membrane and initial segmented cell are saved in `./ResultCell/BothWithRandomnet` and
    `BothWithRandomnetPostseg`, respectively. To unify the label of cell based on AceTree file,
    run 
    ```buildoutcfg
    python shape_analysis.py ./ConfigMemb/shape_config.txt
    ```
* **Structure of folders**: (Folders and files in `.gitignore` are not shown in this repository)
    ```buildoutcfg
    DMapNet is used to segmented membrane stack of C. elegans at cellular level
    DMapNet/
      |--configmemb/: parameters for training, testing and unifying label
      |--Data/: raw membrane, raw nucleus and AceTree file (CD**.csv)
          |--MembTraining/: image data with manual annotations
          |--MembValidation/: image data to be segmented
      |--ModelCell/: trained models 
      |--ResultCell/: Segmentation result
          |--BothWithRandomnet/: Binary membrane segmentation from DMapNet
          |--BothWithRandomnetPostseg/: segmented cell before and after label unifying
          |--NucleusLoc/: nucleus location information and annotation
          |--StatShape/: cell lineage tree (with time duration)
      |--ShapeUtil/: utils for unifying cells and calculating robustness
          |--AceForLabel/: multiple AceTree files for generating namedictionary
          |--RobustStat/: nucleus lost sration and cell surface...
          |--TemCellGraph/: temporary result for calculating surface, volume...
        
      |--Util/: utils for training and testing
    ```
    Result folders will be automatically built.
Codes for the normalization (e.g., resize, rotation) on the segmentation results are available at [CShaperPost](https://github.com/cao13jf/CShaperPost).
### Related
* Project file for CellProfiler involved in evaluation ([link](https://portland-my.sharepoint.com/:u:/g/personal/jfcao3-c_ad_cityu_edu_hk/ETN3Z6j4TklAko6NvQDIujwBwzoixX26EajSOaoeeme2jg?download=1)).
* Parameter files for RACE ([link](https://portland-my.sharepoint.com/:u:/g/personal/jfcao3-c_ad_cityu_edu_hk/EX_iCNByGBtMlZI7G8bRgSMBqNfaCdAbq3cHDrGc-k6d5Q?download=1)). 


### Acknowledgements
* [brats17](https://github.com/taigw/brats17);
* [niftynet](https://niftynet.io).

### Contact
jfcao3-c(at)my.cityu.edu.hk
