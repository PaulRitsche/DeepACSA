from tkinter import *
from tkinter.tix import *
from tkinter import ttk, filedialog
from predict_muscle_area import calculate_batch, calculate_batch_efov


class DeepACSA:
    def __init__(self, root):

        root.title("DeepACSA")

        main = ttk.Frame(root, padding="10 10 12 12")
        main.grid(column=0, row=0, sticky=(N, W, E, S))
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)

        # Tooltips
        tip = Balloon(root)
        model_tipr = Balloon(root)
        flag_tip = Balloon(root)
        scaling_tip = Balloon(root)
        muscle_tip = Balloon(root)
        depth_tip = Balloon(root)
        spacing_tip = Balloon(root)

        # Paths
        # Input directory
        self.input = StringVar()
        input_entry = ttk.Entry(main, width=14, textvariable=self.input)
        input_entry.grid(column=2, row=1, sticky=(W, E))
        # Model path
        self.model = StringVar()
        model_entry = ttk.Entry(main, width=14, textvariable=self.model)
        model_entry.grid(column=2, row=2, sticky=(W, E))
        # Flag path
        self.flags = StringVar()
        flags_entry = ttk.Entry(main, width=14, textvariable=self.flags)
        flags_entry.grid(column=2, row=3, sticky=(W, E))

        # Comboboxes
        # Image Type
        self.scaling = StringVar()
        scaling = ("EFOV", "Static", "Manual")
        scaling_entry = ttk.Combobox(main, width=7, textvariable=self.scaling)
        scaling_entry["values"] = scaling
        scaling_entry["state"] = "readonly"
        scaling_entry.grid(column=2, row=4, sticky=W)
        tip.bind_widget(scaling_entry, 
                        balloonmsg="Choose Image Type from Dropdown List")
        # Muscles
        self.muscle = StringVar()
        muscle = ("VL", "RF", "GM/GL")
        muscle_entry = ttk.Combobox(main, width=7, textvariable=self.muscle)
        muscle_entry["values"] = muscle
        muscle_entry["state"] = "readonly"
        muscle_entry.grid(column=2, row=6, sticky=(W))
        tip.bind_widget(muscle_entry, 
                        balloonmsg="Choose Muscle from Dropdown List")
        # Image Depth
        self.depth = StringVar()
        depth = (2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7)
        depth_entry = ttk.Combobox(main, width=7, textvariable=self.depth)
        depth_entry["values"] = depth
        depth_entry["state"] = "readonly"
        depth_entry.grid(column=2, row=7, sticky=(W))
        tip.bind_widget(depth_entry, 
                        balloonmsg="Choose Image Depth from Dropdown List")
        # Spacing
        self.spacing = StringVar()
        spacing = (5, 10, 15, 20)
        spacing_entry = ttk.Combobox(main, width=7, textvariable=self.spacing)
        spacing_entry["values"] = spacing
        spacing_entry["state"] = "readonly"
        spacing_entry.grid(column=2, row=8, sticky=(W))
        tip.bind_widget(spacing_entry, 
                        balloonmsg="Choose Disance of Scaling Bars" +
                                   " in Image form Dropdown List")

        # Buttons
        # Input directory
        input_button = ttk.Button(main, text="Input",
                                  command=self.get_root_dir)
        input_button.grid(column=3, row=1, sticky=E)
        tip.bind_widget(input_button, 
                        balloonmsg="Choose Root Directory")
        # Model path
        model_button = ttk.Button(main, text="Model",
                                  command=self.get_model_path)
        model_button.grid(column=3, row=2, sticky=E)
        tip.bind_widget(model_button, 
                        balloonmsg="Choose Model Path")
        # Flip Flag path
        flags_button = ttk.Button(main, text="Flip Flag",
                                  command=self.get_flag_dir)
        flags_button.grid(column=3, row=3, sticky=E)
        tip.bind_widget(flags_button, 
                        balloonmsg="Choose Flag File Path")
        # Break Button
        break_button = ttk.Button(main, text="Break", command=self)
        break_button.grid(column=1, row=9, sticky=W)
        # Run Button
        run_button = ttk.Button(main, text="Run", command=self.run_code)
        run_button.grid(column=2, row=9, sticky=(W, E))

        # Labels
        ttk.Label(main, text="Root Directory").grid(column=1, row=1, sticky=W)
        ttk.Label(main, text="Model Path").grid(column=1, row=2, sticky=W)
        ttk.Label(main, text="Flip Flag Path").grid(column=1, row=3, sticky=W)
        ttk.Label(main, text="Image Type").grid(column=1, row=4, sticky=W)
        ttk.Label(main, text="Muscle").grid(column=1, row=6, sticky=W)
        ttk.Label(main, text="Depth (cm)").grid(column=1, row=7, sticky=W)
        ttk.Label(main, text="Spacing (mm)").grid(column=1, row=8, sticky=W)

        for child in main.winfo_children():
            child.grid_configure(padx=5, pady=5)

        # depth_entry.focus()
        root.bind("<Return>", self.run_code)  # execute by pressing return

    def get_root_dir(self):

        root_dir = filedialog.askdirectory()
        self.input.set(root_dir)
        return root_dir

    def get_model_path(self):

        model_dir = filedialog.askopenfilename()
        self.model.set(model_dir)
        return model_dir

    def get_flag_dir(self):

        flag_dir = filedialog.askopenfilename()
        self.flags.set(flag_dir)
        return flag_dir

    def run_code(self):

        selected_muscle = self.muscle.get()
        selected_depth = self.depth.get()
        selected_spacing = self.spacing.get()
        selected_scaling = self.scaling.get()
        selected_input_dir = self.input.get()
        selected_model_path = self.model.get()
        selected_flag_path = self.flags.get()

        if selected_scaling == "EFOV":
            calculate_batch_efov(
                selected_input_dir,
                selected_model_path,
                selected_depth,
                selected_muscle    
                )

        else:
            calculate_batch(
                selected_input_dir,
                selected_flag_path,
                selected_model_path,
                selected_depth,
                selected_spacing,
                selected_muscle,
                selected_scaling
                )

    # def do_break(self):


root = Tk()
DeepACSA(root)
root.mainloop()
