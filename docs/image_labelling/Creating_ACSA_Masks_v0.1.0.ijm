//Project: DeepACSA
//Titel: Create ACSA Masks
//Author: Paul Ritsche
//Last edited: 13.10.22

//////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////USAGE/////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////
/////1. Create folders for output images and masks.///////////////////////////////////////////////////////
/////	There should be a total of two folder, one labeled "train_images", one labeled "train_masks"./////
/////	This is necessary to ensure similar naming of images and corresponding masks./////////////////////
/////2. Run script.///////////////////////////////////////////////////////////////////////////////////////
/////3. Select input folder with images you want to analyze.//////////////////////////////////////////////
/////4. Select previously created output folder for images.///////////////////////////////////////////////
/////5. Select previously created output folder for masks. ///////////////////////////////////////////////
/////6. Scale the image by drawing a scaling line with length of 1cm /////////////////////////////////////
/////7. Create ACSA outline.//////////////////////////////////////////////////////////////////////////////
/////8. Check if mask ASCA is white and background is black.//////////////////////////////////////////////
/////	If not, comment out line 65///////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////


//Specify input dir
input = getDir("Choose input dir");
//Specify output dir for images
output_imgs = getDir("Choose img output dir");
//Specify output dir for masks
output_masks = getDir("Choose mask output dir"); 

//Specify muscle name
muscle = 
"rf_img_"

//Get all files
filelist = getFileList(input);
filelist_output = getFileList(output_imgs);

for (i = 0; i < lengthOf(filelist); i++) {
    if (endsWith(filelist[i], ".tif")) { 
        open(input + File.separator + filelist[i]);

		
		//Change input name
		starting_index = lengthOf(filelist_output);
		save(output_imgs + File.separator + muscle + (i + starting_index)); //MUST be equal to name of mask in line 68!!
		
		//This commented section would be used to delete label artefacts on the US image
		//Delete %Value (use only in esaote images)
		//setTool("rectangle");
        //waitForUser("Delete artefacts", "click OK when done");
        
        //Set measurement parameters
        run("Set Measurements...", "area mean display redirect=None decimal=3");
        
        //Scale image
 
        //setTool("line");
        //waitForUser("Specify scaling line 1cm.");
        //getLine(x1, y1, x2, y2, lineWidth);
	    //line_length = sqrt(pow(x1-x2, 2) + pow(y1-y2, 2));
        //run("Set Scale...", "known=1 unit=cm");
        
        //Create Outline
        setForegroundColor(255, 255, 255);
        setTool("polygon");
        waitForUser("Select Area", "click OK when done");
        setLineWidth(1);
        roiManager("Add");
        
        // measure captured structures	
		roiManager("measure")
		roiManager("Draw");
		roiManager("delete");

		//Create Mask
        run("8-bit");
		setAutoThreshold("Default dark");
		setThreshold(255, 255);
		setOption("BlackBackground", true);
		run("Convert to Mask");
		run("Fill Holes");
		run("Analyze Particles...", "size=100-Infinity pixel show=Masks");
		run("Invert");

		//Save Image
		save(output_masks + File.separator + muscle + (i + starting_index)); //MUST be equal to name of image in line 39!!
		close("*");

    } 
    
}
