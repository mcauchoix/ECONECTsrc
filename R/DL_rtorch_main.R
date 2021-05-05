install.packages("rTorch")
library("rTorch")
rTorch:::install_conda(package="pytorch=1.4", envname="r-torch", 
                       conda="auto", conda_python_version = "3.6", pip=FALSE, 
                       channel="pytorch", 
                       extra_packages=c("torchvision", 
                                        "cpuonly", 
                                        "matplotlib", 
                                        "pandas"))