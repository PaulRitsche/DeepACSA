from tkinter import * 
from tkinter import ttk, filedialog

#from predict_muscle_area import calculate_batch, calculate_batch_efov



class DeepACSA:

    def __init__(self, root):

        root.title("DeepACSA")

        mainframe = ttk.Frame(root, padding="10 10 12 12")
        mainframe.grid(column=0, row=0, sticky=(N, W, E, S))
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)
        
        self.input = StringVar()
        input_entry = ttk.Entry(mainframe, width=14, textvariable=self.input)
        input_entry.grid(column=2, row=1, sticky=(W, E))

        self.model = StringVar()
        model_entry = ttk.Entry(mainframe, width=14, textvariable=self.model)
        model_entry.grid(column=2, row=2, sticky=(W, E))

        self.flags = StringVar()
        flags_entry = ttk.Entry(mainframe, width=14, textvariable=self.flags)
        flags_entry.grid(column=2, row=3, sticky=(W, E))


        self.depth = StringVar()
        depth_entry = ttk.Entry(mainframe, width=7, textvariable=self.depth) #textvariables are kind of globa variables
        depth_entry.grid(column=2, row=5, sticky=(W, E))

        self.spacing = StringVar()
        spacing_entry = ttk.Entry(mainframe, width=7, textvariable=self.spacing)
        spacing_entry.grid(column=2, row=6, sticky=(W, E))
        
        ## Buttons 
        # Input directory
        input_button = ttk.Button(mainframe, text="Input", command=self.get_root_dir)
        input_button.grid(column=3, row=1, sticky=E)
        # Model path
        model_button = ttk.Button(mainframe, text="Model", command=self.get_model_path)
        model_button.grid(column=3, row=2, sticky=E)
        # Flip Flag path
        flags_button = ttk.Button(mainframe, text="Flip Flag", command=self.get_flag_dir)
        flags_button.grid(column=3, row=3, sticky=E)
        # Break Button
        break_button = ttk.Button(mainframe, text="Break")
        break_button.grid(column=1, row=7, sticky=W)
        # Run Button
        run_button = ttk.Button(mainframe, text="Run", command=self.get_root_dir)
        run_button.grid(column=2, row=7, sticky=(W, E))

        ttk.Label(mainframe, text="Root Directory").grid(column=1, row=1, sticky=W)
        ttk.Label(mainframe, text="Model Path").grid(column=1, row=2, sticky=W)
        ttk.Label(mainframe, text="Flip Flag Path").grid(column=1, row=3, sticky=W)
        ttk.Label(mainframe, text="Depth (cm)").grid(column=1, row=5, sticky=W)
        ttk.Label(mainframe, text="Spacing (mm)").grid(column=1, row=6, sticky=W)

        for child in mainframe.winfo_children(): 
            child.grid_configure(padx=5, pady=5)

        depth_entry.focus()
        root.bind("<Return>", self.get_root_dir) #when return is pressed, same as button
        


    def get_root_dir(self):

        root_dir = filedialog.askdirectory()
        self.input.set(root_dir)
        return root_dir

    def get_model_path(self): 

        model_dir = filedialog.askdirectory()
        self.model.set(model_dir)
        return model_dir

    def get_flag_dir(self): 

        flag_dir = filedialog.askdirectory()
        self.flags.set(flag_dir)
        return flag_dir

    # def do_break(self): 



root = Tk()
DeepACSA(root)
root.mainloop()