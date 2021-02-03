input = getDir("Choose input dir");
output = getDir("Choose output dir"); 

filelist = getFileList(input);
for (i = 0; i < lengthOf(filelist); i++) {
    if (endsWith(filelist[i], ".tif")) { 
        open(input + File.separator + filelist[i]);

		save(input + File.separator + "rf_" + (0+i));
        //Create Outline
        setTool("polygon");
        waitForUser("Select Area", "click OK when done");
        setLineWidth(1);
        roiManager("Add");
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
        save(output + File.separator + "rf_" + (0+i));
		close();

    } 
    
}