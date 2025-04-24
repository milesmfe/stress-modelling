import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from sklearn.preprocessing import StandardScaler
import pickle
import pandas as pd


LABEL_MAP = {
    '0': 'Transient',
    '1': 'Baseline',
    '2': 'Stress',
    '3': 'Amusement',
    '4': 'Meditation',
    '5': 'Ignore',
    '6': 'Ignore',
    '7': 'Ignore'
}


class TimeframeGrapherApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🧠 Timeframe Grapher")
        self.root.geometry("900x600")
        self.root.resizable(True, True)
        self.root.protocol("WM_DELETE_WINDOW", self.root.quit)
        self.root.bind("<Control-q>", lambda _: self.root.quit())

        self.data_dict = {}
        self.loaded_data = None
        self.labels = None

        self._build_ui()

    def _build_ui(self):
        self.main_frame = tk.Frame(self.root, padx=10, pady=10)
        self.main_frame.pack(fill='both', expand=True)

        self._build_control_frame()
        self._build_feature_listbox()
        self._build_status_frame()

    def _build_control_frame(self):
        control_frame = tk.Frame(self.main_frame)
        control_frame.pack(anchor='w', fill='x', pady=(0, 10))

        self.load_button = ttk.Button(control_frame, text="📂 Load .pkl", command=self.load_file)
        self.load_button.pack(side='left', padx=(0, 10))

        self.unload_button = ttk.Button(control_frame, text="❌ Unload", command=self.unload_file, state='disabled')
        self.unload_button.pack(side='left', padx=(0, 10))

        self.dropdown_frame = tk.Frame(control_frame)
        self.dropdown_frame.pack(side='left', padx=(10, 10))
        self.dropdown_frame.pack_forget()

        self.subject_selector = ttk.Combobox(self.dropdown_frame, state='readonly')
        self.subject_selector.bind("<<ComboboxSelected>>", self.on_subject_selected)
        self.subject_selector.pack()

        self.overlay_var = tk.BooleanVar()
        self.overlay_check = ttk.Checkbutton(control_frame, text="Overlay Labels", variable=self.overlay_var)
        self.overlay_check.pack(side='left', padx=(20, 0))

        self.plot_button = ttk.Button(control_frame, text="📈 Plot Selected Features", command=self.plot_features, state='disabled')
        self.plot_button.pack(side='right', padx=(0, 10))

        self.export_button = ttk.Button(control_frame, text="📤 Export to CSV", command=self.export_csv, state='disabled')
        self.export_button.pack(side='right', padx=(0, 10))

    def _build_feature_listbox(self):
        self.feature_listbox = tk.Listbox(self.main_frame, selectmode='multiple', width=100, height=20)
        self.feature_listbox.pack(pady=10, fill='x')

    def _build_status_frame(self):
        status_frame = tk.Frame(self.main_frame)
        status_frame.pack(fill='x')

        self.status_label = tk.Label(status_frame, text="Idle", anchor='w')
        self.status_label.pack(side='left', padx=5)

    def load_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("Pickle Files", "*.pkl")])
        if not file_path:
            return

        self._try_load_file(file_path)

    def _try_load_file(self, file_path):
        try:
            self.status_label.config(text="🔄 Loading...")
            with open(file_path, 'rb') as f:
                self.data_dict = pickle.load(f)

            self.populate_subject_selector(list(self.data_dict.keys()))
            self.unload_button.config(state='normal')
        except Exception as e:
            self._handle_load_error(e)

    def _handle_load_error(self, error):
        messagebox.showerror("Loading Failed", f"Error loading pickle file:\n{error}")
        self.data_dict = {}
        self.subject_selector.set('')
        self.dropdown_frame.pack_forget()
        self.status_label.config(text="❌ Load failed")

    def populate_subject_selector(self, subjects):
        self.subject_selector['values'] = subjects
        self.subject_selector.set("Select Subject")
        self.dropdown_frame.pack(side='left', padx=(10, 10))
        self.status_label.config(text="✅ File Loaded — Select a Subject")

    def on_subject_selected(self, event=None):
        subject = self.subject_selector.get()
        if subject in self.data_dict:
            self._load_subject_data(subject)

    def _load_subject_data(self, subject):
        X, y = self.data_dict[subject]
        scalar = StandardScaler()
        self.loaded_data = pd.DataFrame(scalar.fit_transform(X), columns=X.columns)
        self.labels = y

        self.feature_listbox.delete(0, tk.END)
        for col in X.columns:
            self.feature_listbox.insert(tk.END, col)

        self.status_label.config(text=f"✅ Ready — Subject: {subject}")
        self.plot_button.config(state='normal')
        self.export_button.config(state='normal')

    def unload_file(self):
        self._reset_state()
        self.status_label.config(text="🗃️ No file loaded")

    def _reset_state(self):
        self.data_dict = {}
        self.loaded_data = None
        self.labels = None

        self.feature_listbox.delete(0, tk.END)
        self.subject_selector.set('')
        self.subject_selector['values'] = []
        self.dropdown_frame.pack_forget()

        self.plot_button.config(state='disabled')
        self.export_button.config(state='disabled')
        self.unload_button.config(state='disabled')

    def plot_features(self):
        selected_indices = self.feature_listbox.curselection()
        if not selected_indices or self.loaded_data is None:
            messagebox.showwarning("No Features", "Please select features to plot.")
            return

        selected_features = [self.feature_listbox.get(i) for i in selected_indices]
        self._plot_selected_features(selected_features)

    def _plot_selected_features(self, selected_features):
        df = self.loaded_data[selected_features].copy()
        labels = self.labels.copy()
        overlay = self.overlay_var.get()

        fig, ax = plt.subplots(figsize=(14, 6))
        time = range(len(df))

        for feature in selected_features:
            ax.plot(time, df[feature], label=feature, linewidth=1.5)

        if overlay and labels is not None:
            self._add_overlay(ax, labels)

        formatted_features = [f.replace("_", " ").upper() for f in selected_features]
        feature_list = ", ".join(formatted_features)

        ax.legend(loc='upper right')
        ax.set_title(f"{feature_list} Over Time")
        ax.set_xlabel("Window Index")
        ax.set_ylabel("Feature Value")
        plt.tight_layout()
        plt.show()

    def _add_overlay(self, ax, labels):
        unique_labels = sorted(labels.unique())
        cmap = plt.get_cmap('tab10', len(unique_labels))
        label_to_color = {label: cmap(i) for i, label in enumerate(unique_labels)}

        added_labels = set()
        prev_label = labels.iloc[0]
        start = 0

        for i, label in enumerate(labels):
            is_last = i == len(labels) - 1
            if label != prev_label or is_last:
                end = i if label != prev_label else i + 1
                label_str = str(prev_label)
                if label_str not in ['5', '6', '7']:  # Skip ignored
                    y_min, y_max = ax.get_ylim()
                    rect_label = LABEL_MAP.get(label_str, f"Label {label_str}")
                    show_label = rect_label not in added_labels

                    rect = Rectangle(
                        (start, y_min),
                        end - start,
                        y_max - y_min,
                        color=label_to_color[prev_label],
                        alpha=0.2,
                        label=rect_label if show_label else None,
                        zorder=0
                    )
                    ax.add_patch(rect)

                    if show_label:
                        added_labels.add(rect_label)

                start = i
                prev_label = label

        # Ensure both line and overlay legends show up
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(handles, labels, loc='upper right')

    def export_csv(self):
        if self.loaded_data is None or self.labels is None:
            messagebox.showwarning("Nothing to Export", "Please load a subject and select features.")
            return

        filepath = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")],
            title="Save CSV"
        )

        if filepath:
            self._save_to_csv(filepath)

    def _save_to_csv(self, filepath):
        try:
            df = self.loaded_data.copy()
            df["label"] = self.labels.values
            df.to_csv(filepath, index=False)
            messagebox.showinfo("Export Complete", f"Data exported to:\n{filepath}")
        except Exception as e:
            messagebox.showerror("Export Failed", f"Could not save file:\n{e}")


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("TkAgg")

    root = tk.Tk()
    app = TimeframeGrapherApp(root)
    root.mainloop()