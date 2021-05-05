# read
pred=read.csv2("Documents/ECONECT/data/predictionBirdsV1.csv")
lab=read.csv2("Documents/ECONECT/data/list_of_images.csv")

# merge 2 files
lab$image=lab$imageName
d=merge(lab,pred,by="image")
write.csv2()

# recode esp??ces
unique(d$esp.ces)
d$espPred[d$esp.ces=="M\314\251sange charbonni\314\254re"]="PARMAJ"
d$espPred[d$esp.ces=="M\314\251sange bleue"]="PARCAE"

# table
table(d$Species)
# accuracy
d$acc=d$Species==d$espPred
mean(d$acc[d$Species=="PARMAJ"],na.rm=T)
mean(d$acc[d$Species!="PARMAJ"],na.rm=T)
mean(d$acc[d$Species=="PARCAE"],na.rm=T)

# ?? la main
sum(d$esp.ces=="M\314\251sange charbonni\314\254re"&d$Species=="PARMAJ"&d$Size>0.05)/sum(d$Species=="PARMAJ"&d$Size>0.05,na.rm = T)

sum(d$esp.ces=="M\314\251sange charbonni\314\254re"&d$Species=="PARMAJ")/sum(d$Species=="PARMAJ",na.rm = T)


sum(d$esp.ces=="M\314\251sange bleue"&d$Species=="PARCAE"&d$Size>0.05)/sum(d$Species=="PARCAE"&d$Size>0.05,na.rm = T)

sum(d$esp.ces=="M\314\251sange bleue"&d$Species=="PARMAJ"&d$Size>0.05)/sum(d$Species=="PARMAJ")

