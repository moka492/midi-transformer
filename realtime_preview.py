# Real-time piano roll preview for music generation, Shows notes as they are being generated

import tkinter as tk
from tkinter import ttk
import threading
import queue
import time
from collections import deque


class PianoRollPreview:
  
    
    def __init__(self, parent=None):
       
        self.window = tk.Toplevel(parent) if parent else tk.Tk()
        self.window.title(" Real-time Preview")
        self.window.geometry("900x500")
        self.window.configure(bg='#1a1a2e')
        
       
        self.bg_color = '#1a1a2e'
        self.canvas_bg = '#0f1419'
        self.grid_color = '#2a2a3e'
        self.note_color = '#00d9ff'
        self.note_melody = '#00d9ff' 
        self.note_bass = '#ff6b6b'     
        self.text_color = '#eaeaea'
        
    
        self.note_queue = queue.Queue()
        self.is_active = True
        self.notes = deque(maxlen=200) 
        
       
        self.min_pitch = 24  
        self.max_pitch = 108  
        self.pitch_range = self.max_pitch - self.min_pitch
        
     
        self.time_window = 8.0  
        self.current_time = 0.0
        
      
        self._setup_ui()
        
       
        self.window.after(50, self._update_loop)
        
      
        self.window.protocol("WM_DELETE_WINDOW", self._on_close)
    
    def _setup_ui(self):
        
        
        title_frame = tk.Frame(self.window, bg=self.bg_color)
        title_frame.pack(fill='x', padx=20, pady=10)
        
        tk.Label(title_frame, text=" Real-time Generation Preview",
                font=('Segoe UI', 14, 'bold'),
                bg=self.bg_color, fg=self.text_color).pack(side='left')
        
        self.status_label = tk.Label(title_frame, text="Ready",
                                     font=('Segoe UI', 10),
                                     bg=self.bg_color, fg='#a0a0a0')
        self.status_label.pack(side='right')
       
        canvas_frame = tk.Frame(self.window, bg=self.grid_color, padx=2, pady=2)
        canvas_frame.pack(fill='both', expand=True, padx=20, pady=10)
        
        
        self.canvas = tk.Canvas(canvas_frame, bg=self.canvas_bg,
                               highlightthickness=0)
        self.canvas.pack(fill='both', expand=True)
        
      
        info_frame = tk.Frame(self.window, bg=self.bg_color)
        info_frame.pack(fill='x', padx=20, pady=(0, 10))
        
        self.info_label = tk.Label(info_frame, 
                                   text="Waiting for generation...",
                                   font=('Segoe UI', 10),
                                   bg=self.bg_color, fg='#a0a0a0')
        self.info_label.pack()
        
        
        legend_frame = tk.Frame(info_frame, bg=self.bg_color)
        legend_frame.pack(pady=5)
        
       
        tk.Canvas(legend_frame, width=20, height=12, bg=self.note_melody,
                 highlightthickness=0).pack(side='left', padx=5)
        tk.Label(legend_frame, text="Melody", font=('Segoe UI', 9),
                bg=self.bg_color, fg='#a0a0a0').pack(side='left', padx=5)
        
       
        tk.Canvas(legend_frame, width=20, height=12, bg=self.note_bass,
                 highlightthickness=0).pack(side='left', padx=15)
        tk.Label(legend_frame, text="Bass", font=('Segoe UI', 9),
                bg=self.bg_color, fg='#a0a0a0').pack(side='left', padx=5)
    
    def add_note(self, pitch, start_time, duration, track='melody'):
     
        self.note_queue.put({
            'pitch': pitch,
            'start': start_time,
            'duration': duration,
            'track': track
        })
    
    def update_status(self, text):
     
        self.note_queue.put({'status': text})
    
    def set_info(self, text):
        
        self.note_queue.put({'info': text})
    
    def _update_loop(self):
    
        if not self.is_active:
            return
        
       
        try:
            while True:
                item = self.note_queue.get_nowait()
                
                if 'status' in item:
                    self.status_label.config(text=item['status'])
                elif 'info' in item:
                    self.info_label.config(text=item['info'])
                elif 'pitch' in item:
                    self.notes.append(item)
                   
                    note_end = item['start'] + item['duration']
                    if note_end > self.current_time:
                        self.current_time = note_end
        except queue.Empty:
            pass
        
      
        self._draw_piano_roll()
        
       
        self.window.after(50, self._update_loop)
    
    def _draw_piano_roll(self):
        
        self.canvas.delete('all')
        
        width = self.canvas.winfo_width()
        height = self.canvas.winfo_height()
        
        if width < 10 or height < 10:
            return
        
       
        pitch_height = height / self.pitch_range
        
      
        time_end = max(self.current_time, self.time_window)
        time_start = max(0, time_end - self.time_window)
        time_range = time_end - time_start
        
        if time_range == 0:
            time_range = self.time_window
        
      
        for i in range(0, self.pitch_range + 1, 12):  
            y = height - (i * pitch_height)
            self.canvas.create_line(0, y, width, y,
                                   fill=self.grid_color, width=1)
        
       
        beats_in_window = int(time_range) + 1
        for beat in range(beats_in_window):
            time_pos = time_start + beat
            if time_pos <= time_end:
                x = ((time_pos - time_start) / time_range) * width
                self.canvas.create_line(x, 0, x, height,
                                       fill=self.grid_color, width=1, dash=(2, 4))
        
        
        for note in self.notes:
            pitch = note['pitch']
            start = note['start']
            duration = note['duration']
            track = note['track']
            
            
            if start + duration < time_start or start > time_end:
                continue
            
           
            if self.min_pitch <= pitch <= self.max_pitch:
                pitch_idx = pitch - self.min_pitch
                y = height - ((pitch_idx + 1) * pitch_height)
                note_height = pitch_height * 0.8
                
            
                x_start = ((start - time_start) / time_range) * width
                x_end = ((start + duration - time_start) / time_range) * width
                note_width = max(x_end - x_start, 2) 
                
                
                color = self.note_melody if track == 'melody' else self.note_bass
                
                
                self.canvas.create_rectangle(x_start, y, x_start + note_width, y + note_height,
                                            fill=color, outline='', width=0)
        
        
        if self.current_time > 0:
            playhead_x = ((self.current_time - time_start) / time_range) * width
            self.canvas.create_line(playhead_x, 0, playhead_x, height,
                                   fill='#ffffff', width=2)
        
        
        octave_positions = [0, 12, 24, 36, 48, 60, 72] 
        for pos in octave_positions:
            if pos < self.pitch_range:
                y = height - (pos * pitch_height)
                octave = (self.min_pitch + pos) // 12
                self.canvas.create_text(5, y, text=f"C{octave}",
                                       anchor='w', fill='#666',
                                       font=('Segoe UI', 8))
    
    def clear(self):
        
        self.notes.clear()
        self.current_time = 0.0
        self.canvas.delete('all')
    
    def _on_close(self):
   
        self.is_active = False
        self.window.destroy()
    
    def show(self):
      
        self.window.deiconify()
    
    def hide(self):
    
        self.window.withdraw()


def test_preview():
   
    preview = PianoRollPreview()
    preview.update_status("Testing preview")
    preview.set_info("Generating sample notes")
    
   
    def add_test_notes():
        time.sleep(0.5)
        
      
        melody_notes = [60, 62, 64, 65, 67, 69, 71, 72]
        for i, pitch in enumerate(melody_notes):
            preview.add_note(pitch, i * 0.5, 0.4, 'melody')
            time.sleep(0.3)
        
       
        bass_notes = [36, 36, 41, 41, 43, 43, 48, 48]  
        for i, pitch in enumerate(bass_notes):
            preview.add_note(pitch, i * 0.5, 0.8, 'bass')
            time.sleep(0.3)
        
        preview.update_status("Complete!")
        preview.set_info("Test completed - 16 notes generated")
    
    thread = threading.Thread(target=add_test_notes, daemon=True)
    thread.start()
    
    preview.window.mainloop()


if __name__ == "__main__":
    test_preview()
