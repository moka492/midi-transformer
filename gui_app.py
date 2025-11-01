# GUI app for generating and playing MIDI music

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
import random
from pathlib import Path
import subprocess
import platform
import importlib

from generate_musical import generate_musical
import generate_multitrack
from continue_midi import continue_midi
from batch_generate import batch_generate
from realtime_preview import PianoRollPreview


class MusicGeneratorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Music Generator")
        
        importlib.reload(generate_multitrack)
        
        self.root.state('zoomed')
        
        self.root.minsize(700, 800)
        self.root.resizable(True, True)
        
        self.bg_color = '#1a1a2e'
        self.card_bg = '#16213e'
        self.accent = '#0f3460'
        self.primary = '#e94560'
        self.text_white = '#ffffff'
        self.text_gray = '#a8b2d1'
        
        self.root.configure(bg=self.bg_color)
        
        
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('Title.TLabel', font=('Segoe UI', 32, 'bold'), 
                       background=self.bg_color, foreground=self.text_white)
        style.configure('Subtitle.TLabel', font=('Segoe UI', 12), 
                       background=self.bg_color, foreground=self.text_gray)
        style.configure('Label.TLabel', font=('Segoe UI', 11, 'bold'), 
                       background=self.card_bg, foreground=self.text_white)
        style.configure('Value.TLabel', font=('Segoe UI', 11, 'bold'), 
                       background=self.card_bg, foreground=self.primary)
        style.configure('TCombobox', font=('Segoe UI', 11))
        
        self.setup_ui()
        
        self.latest_file = None
        self.is_generating = False
        self.mode = 'generate'  
        self.input_midi = None
        self.preview_window = None  
        
    def setup_ui(self):
        
        title_frame = tk.Frame(self.root, bg=self.bg_color, pady=30)
        title_frame.pack(fill='x')
        
        title = ttk.Label(title_frame, text=" AI Music Generator", 
                         style='Title.TLabel')
        title.pack()
        
        subtitle = ttk.Label(title_frame, 
                            text="Transform AI into beautiful piano compositions", 
                            style='Subtitle.TLabel')
        subtitle.pack(pady=(5, 0))
        
      
        mode_frame = tk.Frame(self.root, bg=self.bg_color)
        mode_frame.pack(pady=(0, 10))
        
        mode_container = tk.Frame(mode_frame, bg=self.accent, pady=2, padx=2)
        mode_container.pack()
        
        self.gen_mode_btn = tk.Button(mode_container, text=" Generate New",
                                      command=lambda: self.switch_mode('generate'),
                                      bg=self.primary, fg='white',
                                      font=('Segoe UI', 10, 'bold'),
                                      relief='flat', cursor='hand2',
                                      padx=20, pady=8, borderwidth=0)
        self.gen_mode_btn.pack(side='left')
        
        self.cont_mode_btn = tk.Button(mode_container, text=" Continue MIDI",
                                       command=lambda: self.switch_mode('continue'),
                                       bg=self.card_bg, fg=self.text_white,
                                       font=('Segoe UI', 10, 'bold'),
                                       relief='flat', cursor='hand2',
                                       padx=20, pady=8, borderwidth=0)
        self.cont_mode_btn.pack(side='left')
        
     
        canvas_frame = tk.Frame(self.root, bg=self.bg_color)
        canvas_frame.pack(fill='both', expand=True, padx=30, pady=(0, 10))
        
        canvas = tk.Canvas(canvas_frame, bg=self.bg_color, highlightthickness=0)
        scrollbar = ttk.Scrollbar(canvas_frame, orient="vertical", command=canvas.yview)
        
    
        scrollable_frame = tk.Frame(canvas, bg=self.bg_color)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        window_id = canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
       
        def _on_canvas_configure(event):
            canvas.itemconfig(window_id, width=event.width)
        canvas.bind("<Configure>", _on_canvas_configure)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
      
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        canvas.bind_all("<MouseWheel>", _on_mousewheel)
        
     
        card = tk.Frame(scrollable_frame, bg=self.card_bg, padx=40, pady=30)
        card.pack(fill='x', padx=0, pady=(0, 20))
        
       
        card.configure(relief='flat', bd=0)
        
      
        self.create_control_group(card, "Musical Scale", 'scale')
        
        
        self.upload_frame = tk.Frame(card, bg=self.card_bg)
        self.upload_frame.pack(fill='x', pady=(0, 20))
        
        upload_label = ttk.Label(self.upload_frame, text="Input MIDI File", 
                                style='Label.TLabel')
        upload_label.pack(anchor='w', pady=(0, 8))
        
        upload_btn_container = tk.Frame(self.upload_frame, bg=self.card_bg)
        upload_btn_container.pack(fill='x')
        
        self.upload_btn = tk.Button(upload_btn_container, text=" Choose MIDI File",
                                    command=self.choose_file,
                                    bg=self.accent, fg='white',
                                    font=('Segoe UI', 10, 'bold'),
                                    relief='flat', cursor='hand2',
                                    pady=10, borderwidth=0, highlightthickness=0)
        self.upload_btn.pack(side='left')
        
        self.file_label = tk.Label(upload_btn_container, text="No file selected",
                                   bg=self.card_bg, fg=self.text_gray,
                                   font=('Segoe UI', 9))
        self.file_label.pack(side='left', padx=15)
        
      
        self.upload_frame.pack_forget()
        
       
        self.create_slider(card, "Length", 8, 32, 16, 'length', " bars")
        
     
        self.create_slider(card, "Creativity", 0.5, 1.0, 0.7, 'temp', "", 0.05)
        
        
        self.create_slider(card, "Chord Density", 0, 80, 40, 'chord', "%", 10)
        
        multitrack_frame = tk.Frame(card, bg=self.card_bg)
        multitrack_frame.pack(fill='x', pady=(10, 20))
        
        self.multitrack_var = tk.BooleanVar(value=False)
        multitrack_check = tk.Checkbutton(multitrack_frame, 
                                         text=" Multi-Track (Separate Melody + Bass)",
                                         variable=self.multitrack_var,
                                         bg=self.card_bg, fg=self.text_white,
                                         selectcolor=self.accent,
                                         activebackground=self.card_bg,
                                         activeforeground=self.primary,
                                         font=('Segoe UI', 11, 'bold'),
                                         cursor='hand2')
        multitrack_check.pack(anchor='w')
        
        multitrack_info = tk.Label(multitrack_frame,
                                  text="Generates separate right hand (melody) and left hand (bass/chords) for more realistic piano music",
                                  bg=self.card_bg, fg=self.text_gray,
                                  font=('Segoe UI', 9),
                                  wraplength=550, justify='left')
        multitrack_info.pack(anchor='w', pady=(5, 0))
        
        
        preview_frame = tk.Frame(card, bg=self.card_bg)
        preview_frame.pack(fill='x', pady=(10, 20))
        
        self.preview_var = tk.BooleanVar(value=True)
        preview_check = tk.Checkbutton(preview_frame, 
                                      text=" Real-Time Preview",
                                      variable=self.preview_var,
                                      bg=self.card_bg, fg=self.text_white,
                                      selectcolor=self.accent,
                                      activebackground=self.card_bg,
                                      activeforeground=self.primary,
                                      font=('Segoe UI', 11, 'bold'),
                                      cursor='hand2')
        preview_check.pack(anchor='w')
        
        preview_info = tk.Label(preview_frame,
                               text="Shows piano roll visualization as notes are being generated",
                               bg=self.card_bg, fg=self.text_gray,
                               font=('Segoe UI', 9),
                               wraplength=550, justify='left')
        preview_info.pack(anchor='w', pady=(5, 0))
        
    
        self.create_slider(card, "Volume", 40, 100, 75, 'volume', "%", 5)
        
        
        btn_frame = tk.Frame(card, bg=self.card_bg)
        btn_frame.pack(fill='x', pady=(30, 10))
        
        self.gen_btn = tk.Button(btn_frame, text=" Generate Music",
                                command=self.generate_music,
                                bg=self.primary, fg='white', 
                                font=('Segoe UI', 14, 'bold'),
                                relief='flat', cursor='hand2',
                                activebackground='#e73855',
                                pady=15, borderwidth=0,
                                highlightthickness=0)
        self.gen_btn.pack(fill='x')
        
    
        status_frame = tk.Frame(card, bg=self.card_bg, pady=10)
        status_frame.pack(fill='x')
        
        self.status_label = tk.Label(status_frame, text="", 
                                     font=('Segoe UI', 10),
                                     bg=self.card_bg, fg=self.text_gray,
                                     wraplength=550)
        self.status_label.pack()
        
    
        action_frame = tk.Frame(card, bg=self.card_bg)
        action_frame.pack(fill='x', pady=(10, 0))
        
      
        btn_container = tk.Frame(action_frame, bg=self.card_bg)
        btn_container.pack()
        
        self.play_btn = tk.Button(btn_container, text=" Play",
                                  command=self.play_music, state='disabled',
                                  bg='#0f3460', fg='white',
                                  font=('Segoe UI', 12, 'bold'),
                                  relief='flat', cursor='hand2',
                                  activebackground='#0d2d50',
                                  pady=12, padx=30, borderwidth=0,
                                  highlightthickness=0)
        self.play_btn.pack(side='left', padx=5)
        
        self.open_btn = tk.Button(btn_container, text="  Open Folder",
                                  command=self.open_folder,
                                  bg='#0f3460', fg='white',
                                  font=('Segoe UI', 12, 'bold'),
                                  relief='flat', cursor='hand2',
                                  activebackground='#0d2d50',
                                  pady=12, padx=30, borderwidth=0,
                                  highlightthickness=0)
        self.open_btn.pack(side='left', padx=5)
        
        self.batch_btn = tk.Button(btn_container, text="  Batch (10x)",
                                   command=self.batch_generate,
                                   bg='#0f3460', fg='white',
                                   font=('Segoe UI', 12, 'bold'),
                                   relief='flat', cursor='hand2',
                                   activebackground='#0d2d50',
                                   pady=12, padx=30, borderwidth=0,
                                   highlightthickness=0)
        self.batch_btn.pack(side='left', padx=5)
        
       
        footer_frame = tk.Frame(scrollable_frame, bg=self.bg_color, pady=15)
        footer_frame.pack(fill='x')
        
        info = ttk.Label(footer_frame, 
                        text="Built by Moksh Saksena • Transformer AI",
                        style='Subtitle.TLabel', justify='center')
        info.pack()
        
    def create_control_group(self, parent, label_text, var_name):
     
        frame = tk.Frame(parent, bg=self.card_bg)
        frame.pack(fill='x', pady=(0, 20))
        
        label = ttk.Label(frame, text=label_text, style='Label.TLabel')
        label.pack(anchor='w', pady=(0, 8))
        
        if var_name == 'scale':
            from music_theory import SCALES
            self.scale_var = tk.StringVar(value='C_major')
            combo = ttk.Combobox(frame, textvariable=self.scale_var,
                               state='readonly', font=('Segoe UI', 11),
                               height=10)
            combo['values'] = tuple(sorted(SCALES.keys()))
            combo.pack(fill='x')
            
    def create_slider(self, parent, label_text, min_val, max_val, default, var_name, suffix="", resolution=1):
     
        frame = tk.Frame(parent, bg=self.card_bg)
        frame.pack(fill='x', pady=(0, 20))
        
      
        header = tk.Frame(frame, bg=self.card_bg)
        header.pack(fill='x', pady=(0, 8))
        
        label = ttk.Label(header, text=label_text, style='Label.TLabel')
        label.pack(side='left')
        
        value_label = ttk.Label(header, text=f"{default}{suffix}", style='Value.TLabel')
        value_label.pack(side='right')
        
    
        if var_name == 'temp':
            var = tk.DoubleVar(value=default)
            self.temp_var = var
            self.temp_value = value_label
        elif var_name == 'length':
            var = tk.IntVar(value=default)
            self.length_var = var
            self.length_value = value_label
        elif var_name == 'chord':
            var = tk.IntVar(value=default)
            self.chord_var = var
            self.chord_value = value_label
        elif var_name == 'volume':
            var = tk.IntVar(value=default)
            self.volume_var = var
            self.volume_value = value_label
            
        slider = tk.Scale(frame, from_=min_val, to=max_val, resolution=resolution,
                         variable=var, orient='horizontal',
                         command=lambda v: self.update_slider_value(var_name, v, suffix),
                         bg=self.card_bg, fg=self.text_white, 
                         highlightthickness=0, showvalue=0,
                         troughcolor=self.accent, 
                         activebackground=self.primary,
                         sliderrelief='flat', bd=0,
                         length=500, width=20)
        slider.pack(fill='x')
        
    def update_slider_value(self, var_name, val, suffix):
       
        if var_name == 'temp':
            self.temp_value.config(text=f"{float(val):.2f}{suffix}")
        elif var_name == 'length':
            self.length_value.config(text=f"{int(float(val))}{suffix}")
        elif var_name == 'chord':
            self.chord_value.config(text=f"{int(float(val))}{suffix}")
        elif var_name == 'volume':
            self.volume_value.config(text=f"{int(float(val))}{suffix}")
    
    def switch_mode(self, mode):
       
        self.mode = mode
        
        if mode == 'generate':
            self.gen_mode_btn.config(bg=self.primary)
            self.cont_mode_btn.config(bg=self.card_bg, fg=self.text_white)
            self.upload_frame.pack_forget()  
            self.gen_btn.config(text=" Generate Music")
        else: 
            self.gen_mode_btn.config(bg=self.card_bg, fg=self.text_white)
            self.cont_mode_btn.config(bg=self.primary, fg='white')
            self.upload_frame.pack(fill='x', pady=(0, 20))  
            self.gen_btn.config(text=" Continue Music")
    
    def choose_file(self):
       
        filename = filedialog.askopenfilename(
            title="Select MIDI File",
            filetypes=[("MIDI files", "*.mid *.midi"), ("All files", "*.*")]
        )
        
        if filename:
            self.input_midi = filename
            self.file_label.config(text=Path(filename).name, fg=self.primary)
        
    def generate_music(self):
        if self.is_generating:
            return
        
     
        if self.mode == 'continue' and not self.input_midi:
            messagebox.showwarning("No File", "please choose a MIDI file to continue")
            return
            
        self.is_generating = True
        self.gen_btn.config(state='disabled', text=" Generating...")
        
        if self.mode == 'generate':
            self.status_label.config(text="Generating music, this may take 30-60 seconds")
        else:
            self.status_label.config(text="Continuing your music, this may take 30-60 seconds")
        
       
        thread = threading.Thread(target=self._generate_thread)
        thread.daemon = True
        thread.start()
        
    def _generate_thread(self):
        try:
            scale = self.scale_var.get()
            length = self.length_var.get()
            temp = self.temp_var.get()
            chord = self.chord_var.get()
            use_multitrack = self.multitrack_var.get()
            use_preview = self.preview_var.get()
            output_dir = Path("decoded")
            
          
            preview_callback = None
            if use_preview:
                self.root.after(0, self._create_preview_window)
             
                import time
                time.sleep(0.5)
                
              
                def preview_callback(pitch, start_time, duration, track):
                    if self.preview_window:
                        self.preview_window.add_note(pitch, start_time, duration, track)
            
            if self.mode == 'generate':
                
                if use_multitrack:
                    if use_preview and self.preview_window:
                        self.preview_window.update_status("Generating multi-track music...")
                        self.preview_window.set_info(f"Scale: {scale} | Temperature: {temp} | Chord Density: {chord}%")
                    
                    generate_multitrack.generate_multitrack(num_samples=1, max_tokens=length * 30, scale=scale,
                                      temperature=temp, chord_density=chord,
                                      preview_callback=preview_callback)
                 
                    files = sorted(output_dir.glob("multitrack_*.mid"), 
                                 key=lambda x: x.stat().st_mtime)
                else:
                    if use_preview and self.preview_window:
                        self.preview_window.update_status("Generating single-track music...")
                        self.preview_window.set_info(f"Scale: {scale} | Temperature: {temp}")
                    
                   
                    generate_musical(n=1, max_tok=length * 30, scale=scale, preview_callback=preview_callback)
                    
                    files = sorted(output_dir.glob("musical_*.mid"), 
                                 key=lambda x: x.stat().st_mtime)
            else:
                if use_preview and self.preview_window:
                    self.preview_window.update_status("Continuing music...")
                    self.preview_window.set_info(f"Adding {length} bars (matching original style)")
                
                
                output_file = output_dir / f"continued_{Path(self.input_midi).stem}.mid"
                continue_midi(self.input_midi, str(output_file), 
                            continue_bars=length, temperature=0.7, 
                            scale='C_major', preview_callback=preview_callback)
                files = [output_file]
            
            if use_preview and self.preview_window:
                self.preview_window.update_status("Complete!")
                self.preview_window.set_info("Generation finished - you can close this window")
            
            if files:
                self.latest_file = files[-1]
                self.root.after(0, self._generation_complete)
            else:
                self.root.after(0, self._generation_error, "No file generated")
                
        except Exception as e:
            self.root.after(0, self._generation_error, str(e))
    
    def _create_preview_window(self):

        if not self.preview_window or not hasattr(self.preview_window, 'is_active') or not self.preview_window.is_active:
            self.preview_window = PianoRollPreview(self.root)
            self.preview_window.update_status("Initializing")
            self.preview_window.set_info("Preparing to generate")
            
    def _generation_complete(self):
        self.is_generating = False
        
        if self.mode == 'generate':
            self.gen_btn.config(state='normal', text=" Generate Music")
            self.status_label.config(text=f"Generated: {self.latest_file.name}")
        else:
            self.gen_btn.config(state='normal', text=" Continue Music")
            self.status_label.config(text=f"Continued: {self.latest_file.name}")
            
        self.play_btn.config(state='normal')
        
    def _generation_error(self, error):
        self.is_generating = False
        self.gen_btn.config(state='normal', text="Generate Music")
        self.status_label.config(text=f" Error: {error}")
        messagebox.showerror("Generation Error", f"Failed to generate music:\n{error}")
    
    def batch_generate(self):
       
        if self.is_generating:
            messagebox.showwarning("Busy", "Generation in progress!")
            return
        
       
        dialog = tk.Toplevel(self.root)
        dialog.title("Batch Generate")
        dialog.geometry("400x250")
        dialog.configure(bg=self.bg_color)
        dialog.transient(self.root)
        dialog.grab_set()
        
    
        dialog.update_idletasks()
        x = self.root.winfo_x() + (self.root.winfo_width() // 2) - (dialog.winfo_width() // 2)
        y = self.root.winfo_y() + (self.root.winfo_height() // 2) - (dialog.winfo_height() // 2)
        dialog.geometry(f"+{x}+{y}")
        
        
        tk.Label(dialog, text=" Batch Generation",
                font=('Segoe UI', 16, 'bold'),
                bg=self.bg_color, fg=self.text_white).pack(pady=20)
        
      
        count_frame = tk.Frame(dialog, bg=self.bg_color)
        count_frame.pack(pady=10)
        
        tk.Label(count_frame, text="Number of samples:",
                font=('Segoe UI', 12),
                bg=self.bg_color, fg=self.text_gray).pack(side='left', padx=10)
        
        count_var = tk.StringVar(value="10")
        count_entry = tk.Entry(count_frame, textvariable=count_var,
                              font=('Segoe UI', 12),
                              width=10,
                              bg=self.card_bg, fg=self.text_white,
                              insertbackground=self.text_white,
                              relief='flat', borderwidth=2)
        count_entry.pack(side='left', padx=10)
        
       
        varied_var = tk.BooleanVar(value=False)
        varied_check = tk.Checkbutton(dialog, text="Randomize parameters",
                                     variable=varied_var,
                                     font=('Segoe UI', 11),
                                     bg=self.bg_color, fg=self.text_gray,
                                     selectcolor=self.card_bg,
                                     activebackground=self.bg_color,
                                     activeforeground=self.text_white)
        varied_check.pack(pady=10)
        
      
        info_label = tk.Label(dialog, 
                             text="(This will use current settings unless randomized)",
                             font=('Segoe UI', 9, 'italic'),
                             bg=self.bg_color, fg=self.text_gray)
        info_label.pack(pady=5)
        
       
        btn_frame = tk.Frame(dialog, bg=self.bg_color)
        btn_frame.pack(pady=20)
        
        def start_batch():
            try:
                count = int(count_var.get())
                if count < 1 or count > 50:
                    messagebox.showerror("Invalid", "Count must be between 1-50")
                    return
                
                dialog.destroy()
                self._run_batch_generation(count, varied_var.get())
            except ValueError:
                messagebox.showerror("Invalid", "Please enter a valid number")
        
        tk.Button(btn_frame, text="Generate",
                 command=start_batch,
                 bg=self.primary, fg='white',
                 font=('Segoe UI', 11, 'bold'),
                 relief='flat', cursor='hand2',
                 activebackground='#e73855',
                 pady=8, padx=30).pack(side='left', padx=5)
        
        tk.Button(btn_frame, text="Cancel",
                 command=dialog.destroy,
                 bg=self.accent, fg='white',
                 font=('Segoe UI', 11),
                 relief='flat', cursor='hand2',
                 activebackground='#0d2d50',
                 pady=8, padx=30).pack(side='left', padx=5)
    
    def _run_batch_generation(self, count, varied):
        
        self.is_generating = True
        self.gen_btn.config(state='disabled')
        self.batch_btn.config(state='disabled')
        self.status_label.config(text=f"Generating batch: 0/{count}")
        
        def progress_callback(current, total):
            self.root.after(0, lambda: self.status_label.config(
                text=f"Generating batch: {current}/{total}"
            ))
        
        def batch_task():
            try:
              
                scale = self.scale_var.get()
                length = self.length_var.get()
                temperature = self.temp_var.get()
                chord_density = self.chord_var.get()
                multitrack = self.multitrack_var.get()
                
                
                if varied:
                    from batch_generate import batch_generate_varied
                    files = batch_generate_varied(
                        count=count,
                        multitrack=multitrack,
                        base_length=length,
                        progress_callback=progress_callback
                    )
                else:
                    files = batch_generate(
                        count=count,
                        multitrack=multitrack,
                        scale=scale,
                        length=length,
                        temperature=temperature,
                        chord_density=chord_density,
                        progress_callback=progress_callback
                    )
                
              
                self.latest_file = Path(files[-1]) if files else None
                self.root.after(0, self._batch_complete, count, varied)
                
            except Exception as e:
                self.root.after(0, self._generation_error, str(e))
        
        thread = threading.Thread(target=batch_task, daemon=True)
        thread.start()
    
    def _batch_complete(self, count, varied):
       
        self.is_generating = False
        self.gen_btn.config(state='normal')
        self.batch_btn.config(state='normal')
        
        folder_name = "batch_varied" if varied else "batch"
        self.status_label.config(text=f" Generated {count} samples in decoded/{folder_name}/")
        
       
        result = messagebox.askyesno("Batch Complete", 
                                     f"Generated {count} samples!\n\nOpen folder?")
        if result:
            folder = Path("decoded") / folder_name
            folder.mkdir(parents=True, exist_ok=True)
            try:
                if platform.system() == 'Windows':
                    subprocess.Popen(['explorer', str(folder)])
                elif platform.system() == 'Darwin':
                    subprocess.Popen(['open', str(folder)])
                else:
                    subprocess.Popen(['xdg-open', str(folder)])
            except Exception as e:
                messagebox.showerror("Error", f"Could not open folder:\n{e}")
        
    def play_music(self):
        if not self.latest_file or not self.latest_file.exists():
            messagebox.showwarning("No File", "No music file to play!")
            return
            
       
        try:
            if platform.system() == 'Windows':
                subprocess.Popen(['start', '', str(self.latest_file)], shell=True)
            elif platform.system() == 'Darwin':  
                subprocess.Popen(['open', str(self.latest_file)])
            else:
                subprocess.Popen(['xdg-open', str(self.latest_file)])
        except Exception as e:
            messagebox.showerror("Play Error", f"Could not open file:\n{e}")
            
    def open_folder(self):
       
        if not self.latest_file or not self.latest_file.exists():
            messagebox.showinfo("No File", "Generate music first!")
            return
            
        try:
            if platform.system() == 'Windows':
               
                subprocess.Popen(['explorer', '/select,', str(self.latest_file)])
            elif platform.system() == 'Darwin':
                
                subprocess.Popen(['open', '-R', str(self.latest_file)])
            else:
                
                subprocess.Popen(['xdg-open', str(self.latest_file.parent)])
        except Exception as e:
            messagebox.showerror("Error", f"Could not open folder:\n{e}")


if __name__ == "__main__":
    root = tk.Tk()
    app = MusicGeneratorApp(root)
    root.mainloop()
