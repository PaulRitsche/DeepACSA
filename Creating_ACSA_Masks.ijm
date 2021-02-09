input = getDir("Choose input dir");
output_imgs = getDir("Choose img output dir");
output_masks = getDir("Choose maskoutput dir"); 

filelist = getFileList(input);
for (i = 0; i < lengthOf(filelist); i++) {
    if (endsWith(filelist[i], ".tif")) { 
        open(input + File.separator + filelist[i]);
		
		//Change input name
		save(output_imgs + File.separator + "rf_" + (51+i));
        //Create Outline
        setTool("polygon");
        waitForUser("Select Area", "click OK when done");
        setLineWidth(1);
        roiManager("Add");
        //roiManager("measure");
		roiManager("Draw");
		roiManager("delete");
		
		//Create Mask
        run("8-bit");
		setAutoThreshold("Default dark");
		setThreshold(255, 255);
		setOption("BlackBackground", true);
		run("Convert to Mask");
		run("Fill Holes");

		//Save Image
        save(output_masks + File.separator + "rf_" + (51+i));
		close();

    } 
    
}