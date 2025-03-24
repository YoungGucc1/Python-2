using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text.Json;
using System.Windows.Forms;

namespace ClipboardManager
{
    public partial class MainForm : Form
    {
        private List<ClipboardItem> clipboardHistory = new List<ClipboardItem>();
        private List<ClipboardItem> favorites = new List<ClipboardItem>();
        private Dictionary<string, Keys> hotkeys = new Dictionary<string, Keys>();
        private GlobalHotkey globalHotkey;
        private int maxHistoryItems = 100;
        private Timer clipboardMonitor;

        public MainForm()
        {
            InitializeComponent();
            InitializeClipboardMonitor();
            LoadSettings();
            SetupTabs();
        }

        private void InitializeComponent()
        {
            this.Text = "Clipboard Manager";
            this.Size = new System.Drawing.Size(800, 600);
            this.StartPosition = FormStartPosition.CenterScreen;
            this.FormClosing += MainForm_FormClosing;

            // Create tab control
            TabControl tabControl = new TabControl();
            tabControl.Dock = DockStyle.Fill;
            this.Controls.Add(tabControl);

            // Create tabs
            TabPage historyTab = new TabPage("History");
            TabPage favoritesTab = new TabPage("Favorites");
            TabPage settingsTab = new TabPage("Settings");

            tabControl.TabPages.Add(historyTab);
            tabControl.TabPages.Add(favoritesTab);
            tabControl.TabPages.Add(settingsTab);

            // Create ListView for history
            ListView historyListView = new ListView();
            historyListView.Dock = DockStyle.Fill;
            historyListView.View = View.Details;
            historyListView.FullRowSelect = true;
            historyListView.Columns.Add("Content", 400);
            historyListView.Columns.Add("Time", 150);
            historyListView.Columns.Add("Type", 100);
            historyListView.ContextMenuStrip = CreateHistoryContextMenu();
            historyListView.DoubleClick += HistoryListView_DoubleClick;
            historyListView.Name = "historyListView";
            historyTab.Controls.Add(historyListView);

            // Create ListView for favorites
            ListView favoritesListView = new ListView();
            favoritesListView.Dock = DockStyle.Fill;
            favoritesListView.View = View.Details;
            favoritesListView.FullRowSelect = true;
            favoritesListView.Columns.Add("Content", 400);
            favoritesListView.Columns.Add("Hotkey", 150);
            favoritesListView.Columns.Add("Type", 100);
            favoritesListView.ContextMenuStrip = CreateFavoritesContextMenu();
            favoritesListView.DoubleClick += FavoritesListView_DoubleClick;
            favoritesListView.Name = "favoritesListView";
            favoritesTab.Controls.Add(favoritesListView);

            // Create settings panel
            Panel settingsPanel = new Panel();
            settingsPanel.Dock = DockStyle.Fill;
            settingsTab.Controls.Add(settingsPanel);

            // Max history items
            Label maxItemsLabel = new Label();
            maxItemsLabel.Text = "Max History Items:";
            maxItemsLabel.Location = new System.Drawing.Point(20, 20);
            maxItemsLabel.AutoSize = true;
            settingsPanel.Controls.Add(maxItemsLabel);

            NumericUpDown maxItemsUpDown = new NumericUpDown();
            maxItemsUpDown.Minimum = 10;
            maxItemsUpDown.Maximum = 1000;
            maxItemsUpDown.Value = maxHistoryItems;
            maxItemsUpDown.Location = new System.Drawing.Point(150, 18);
            maxItemsUpDown.ValueChanged += (s, e) => { maxHistoryItems = (int)maxItemsUpDown.Value; };
            settingsPanel.Controls.Add(maxItemsUpDown);

            // Import/Export buttons
            Button importButton = new Button();
            importButton.Text = "Import Data";
            importButton.Location = new System.Drawing.Point(20, 60);
            importButton.Click += ImportButton_Click;
            settingsPanel.Controls.Add(importButton);

            Button exportButton = new Button();
            exportButton.Text = "Export Data";
            exportButton.Location = new System.Drawing.Point(150, 60);
            exportButton.Click += ExportButton_Click;
            settingsPanel.Controls.Add(exportButton);
        }

        private void InitializeClipboardMonitor()
        {
            clipboardMonitor = new Timer();
            clipboardMonitor.Interval = 500; // Check clipboard every 500ms
            clipboardMonitor.Tick += ClipboardMonitor_Tick;
            clipboardMonitor.Start();

            globalHotkey = new GlobalHotkey(this.Handle);
        }

        private void ClipboardMonitor_Tick(object sender, EventArgs e)
        {
            try
            {
                if (Clipboard.ContainsText())
                {
                    string clipboardText = Clipboard.GetText();
                    
                    // Avoid duplicate entries
                    if (clipboardHistory.Count == 0 || clipboardHistory[0].Content != clipboardText)
                    {
                        ClipboardItem newItem = new ClipboardItem
                        {
                            Content = clipboardText,
                            Timestamp = DateTime.Now,
                            Type = "Text"
                        };

                        clipboardHistory.Insert(0, newItem);
                        
                        // Limit history size
                        if (clipboardHistory.Count > maxHistoryItems)
                        {
                            clipboardHistory.RemoveAt(clipboardHistory.Count - 1);
                        }

                        UpdateHistoryListView();
                    }
                }
            }
            catch (Exception ex)
            {
                // Handle clipboard access errors
                Console.WriteLine($"Clipboard error: {ex.Message}");
            }
        }

        private void SetupTabs()
        {
            UpdateHistoryListView();
            UpdateFavoritesListView();
        }

        private void UpdateHistoryListView()
        {
            ListView historyListView = this.Controls.Find("historyListView", true).FirstOrDefault() as ListView;
            if (historyListView != null)
            {
                historyListView.Items.Clear();
                foreach (var item in clipboardHistory)
                {
                    var listItem = new ListViewItem(item.Content.Length > 50 ? item.Content.Substring(0, 47) + "..." : item.Content);
                    listItem.SubItems.Add(item.Timestamp.ToString());
                    listItem.SubItems.Add(item.Type);
                    listItem.Tag = item;
                    historyListView.Items.Add(listItem);
                }
            }
        }

        private void UpdateFavoritesListView()
        {
            ListView favoritesListView = this.Controls.Find("favoritesListView", true).FirstOrDefault() as ListView;
            if (favoritesListView != null)
            {
                favoritesListView.Items.Clear();
                foreach (var item in favorites)
                {
                    var listItem = new ListViewItem(item.Content.Length > 50 ? item.Content.Substring(0, 47) + "..." : item.Content);
                    string hotkeyText = hotkeys.ContainsKey(item.Id) ? hotkeys[item.Id].ToString() : "None";
                    listItem.SubItems.Add(hotkeyText);
                    listItem.SubItems.Add(item.Type);
                    listItem.Tag = item;
                    favoritesListView.Items.Add(listItem);
                }
            }
        }

        private ContextMenuStrip CreateHistoryContextMenu()
        {
            ContextMenuStrip menu = new ContextMenuStrip();
            
            ToolStripMenuItem copyItem = new ToolStripMenuItem("Copy");
            copyItem.Click += (s, e) => {
                ListView listView = this.Controls.Find("historyListView", true).FirstOrDefault() as ListView;
                if (listView.SelectedItems.Count > 0)
                {
                    ClipboardItem item = listView.SelectedItems[0].Tag as ClipboardItem;
                    Clipboard.SetText(item.Content);
                }
            };
            
            ToolStripMenuItem addToFavoritesItem = new ToolStripMenuItem("Add to Favorites");
            addToFavoritesItem.Click += (s, e) => {
                ListView listView = this.Controls.Find("historyListView", true).FirstOrDefault() as ListView;
                if (listView.SelectedItems.Count > 0)
                {
                    ClipboardItem item = listView.SelectedItems[0].Tag as ClipboardItem;
                    AddToFavorites(item);
                }
            };
            
            ToolStripMenuItem deleteItem = new ToolStripMenuItem("Delete");
            deleteItem.Click += (s, e) => {
                ListView listView = this.Controls.Find("historyListView", true).FirstOrDefault() as ListView;
                if (listView.SelectedItems.Count > 0)
                {
                    ClipboardItem item = listView.SelectedItems[0].Tag as ClipboardItem;
                    clipboardHistory.Remove(item);
                    UpdateHistoryListView();
                }
            };
            
            menu.Items.Add(copyItem);
            menu.Items.Add(addToFavoritesItem);
            menu.Items.Add(deleteItem);
            
            return menu;
        }

        private ContextMenuStrip CreateFavoritesContextMenu()
        {
            ContextMenuStrip menu = new ContextMenuStrip();
            
            ToolStripMenuItem copyItem = new ToolStripMenuItem("Copy");
            copyItem.Click += (s, e) => {
                ListView listView = this.Controls.Find("favoritesListView", true).FirstOrDefault() as ListView;
                if (listView.SelectedItems.Count > 0)
                {
                    ClipboardItem item = listView.SelectedItems[0].Tag as ClipboardItem;
                    Clipboard.SetText(item.Content);
                }
            };
            
            ToolStripMenuItem setHotkeyItem = new ToolStripMenuItem("Set Hotkey");
            setHotkeyItem.Click += (s, e) => {
                ListView listView = this.Controls.Find("favoritesListView", true).FirstOrDefault() as ListView;
                if (listView.SelectedItems.Count > 0)
                {
                    ClipboardItem item = listView.SelectedItems[0].Tag as ClipboardItem;
                    AssignHotkey(item);
                }
            };
            
            ToolStripMenuItem removeItem = new ToolStripMenuItem("Remove from Favorites");
            removeItem.Click += (s, e) => {
                ListView listView = this.Controls.Find("favoritesListView", true).FirstOrDefault() as ListView;
                if (listView.SelectedItems.Count > 0)
                {
                    ClipboardItem item = listView.SelectedItems[0].Tag as ClipboardItem;
                    if (hotkeys.ContainsKey(item.Id))
                    {
                        globalHotkey.UnregisterHotKey(item.Id);
                        hotkeys.Remove(item.Id);
                    }
                    favorites.Remove(item);
                    UpdateFavoritesListView();
                }
            };
            
            menu.Items.Add(copyItem);
            menu.Items.Add(setHotkeyItem);
            menu.Items.Add(removeItem);
            
            return menu;
        }

        private void HistoryListView_DoubleClick(object sender, EventArgs e)
        {
            ListView listView = sender as ListView;
            if (listView.SelectedItems.Count > 0)
            {
                ClipboardItem item = listView.SelectedItems[0].Tag as ClipboardItem;
                Clipboard.SetText(item.Content);
            }
        }

        private void FavoritesListView_DoubleClick(object sender, EventArgs e)
        {
            ListView listView = sender as ListView;
            if (listView.SelectedItems.Count > 0)
            {
                ClipboardItem item = listView.SelectedItems[0].Tag as ClipboardItem;
                Clipboard.SetText(item.Content);
            }
        }

        private void AddToFavorites(ClipboardItem item)
        {
            // Create a new favorite item (clone the history item)
            ClipboardItem favoriteItem = new ClipboardItem
            {
                Id = Guid.NewGuid().ToString(),
                Content = item.Content,
                Timestamp = item.Timestamp,
                Type = item.Type
            };
            
            // Add to favorites if not already present
            if (!favorites.Any(f => f.Content == item.Content))
            {
                favorites.Add(favoriteItem);
                UpdateFavoritesListView();
            }
        }

        private void AssignHotkey(ClipboardItem item)
        {
            using (HotkeyForm hotkeyForm = new HotkeyForm())
            {
                if (hotkeyForm.ShowDialog() == DialogResult.OK)
                {
                    Keys hotkey = hotkeyForm.SelectedHotkey;
                    
                    // If this item already has a hotkey, unregister it
                    if (hotkeys.ContainsKey(item.Id))
                    {
                        globalHotkey.UnregisterHotKey(item.Id);
                    }
                    
                    // Register the new hotkey
                    hotkeys[item.Id] = hotkey;
                    globalHotkey.RegisterHotKey(item.Id, hotkey, () => {
                        Clipboard.SetText(item.Content);
                    });
                    
                    UpdateFavoritesListView();
                }
            }
        }

        private void ImportButton_Click(object sender, EventArgs e)
        {
            using (OpenFileDialog dialog = new OpenFileDialog())
            {
                dialog.Filter = "JSON files (*.json)|*.json";
                if (dialog.ShowDialog() == DialogResult.OK)
                {
                    try
                    {
                        string json = File.ReadAllText(dialog.FileName);
                        ClipboardData data = JsonSerializer.Deserialize<ClipboardData>(json);
                        
                        clipboardHistory = data.History;
                        favorites = data.Favorites;
                        hotkeys = data.Hotkeys;
                        
                        // Re-register hotkeys
                        foreach (var kvp in hotkeys)
                        {
                            string id = kvp.Key;
                            Keys key = kvp.Value;
                            
                            ClipboardItem item = favorites.FirstOrDefault(f => f.Id == id);
                            if (item != null)
                            {
                                globalHotkey.RegisterHotKey(id, key, () => {
                                    Clipboard.SetText(item.Content);
                                });
                            }
                        }
                        
                        UpdateHistoryListView();
                        UpdateFavoritesListView();
                        
                        MessageBox.Show("Data imported successfully.", "Import", MessageBoxButtons.OK, MessageBoxIcon.Information);
                    }
                    catch (Exception ex)
                    {
                        MessageBox.Show($"Error importing data: {ex.Message}", "Import Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
                    }
                }
            }
        }

        private void ExportButton_Click(object sender, EventArgs e)
        {
            using (SaveFileDialog dialog = new SaveFileDialog())
            {
                dialog.Filter = "JSON files (*.json)|*.json";
                if (dialog.ShowDialog() == DialogResult.OK)
                {
                    try
                    {
                        ClipboardData data = new ClipboardData
                        {
                            History = clipboardHistory,
                            Favorites = favorites,
                            Hotkeys = hotkeys
                        };
                        
                        string json = JsonSerializer.Serialize(data, new JsonSerializerOptions { WriteIndented = true });
                        File.WriteAllText(dialog.FileName, json);
                        
                        MessageBox.Show("Data exported successfully.", "Export", MessageBoxButtons.OK, MessageBoxIcon.Information);
                    }
                    catch (Exception ex)
                    {
                        MessageBox.Show($"Error exporting data: {ex.Message}", "Export Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
                    }
                }
            }
        }

        private void LoadSettings()
        {
            string appDataPath = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData), "ClipboardManager");
            string settingsFile = Path.Combine(appDataPath, "settings.json");
            
            if (File.Exists(settingsFile))
            {
                try
                {
                    string json = File.ReadAllText(settingsFile);
                    ClipboardData data = JsonSerializer.Deserialize<ClipboardData>(json);
                    
                    clipboardHistory = data.History;
                    favorites = data.Favorites;
                    hotkeys = data.Hotkeys;
                    
                    // Re-register hotkeys
                    foreach (var kvp in hotkeys)
                    {
                        string id = kvp.Key;
                        Keys key = kvp.Value;
                        
                        ClipboardItem item = favorites.FirstOrDefault(f => f.Id == id);
                        if (item != null)
                        {
                            globalHotkey.RegisterHotKey(id, key, () => {
                                Clipboard.SetText(item.Content);
                            });
                        }
                    }
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"Error loading settings: {ex.Message}");
                }
            }
        }

        private void SaveSettings()
        {
            string appDataPath = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData), "ClipboardManager");
            
            if (!Directory.Exists(appDataPath))
            {
                Directory.CreateDirectory(appDataPath);
            }
            
            string settingsFile = Path.Combine(appDataPath, "settings.json");
            
            try
            {
                ClipboardData data = new ClipboardData
                {
                    History = clipboardHistory,
                    Favorites = favorites,
                    Hotkeys = hotkeys
                };
                
                string json = JsonSerializer.Serialize(data, new JsonSerializerOptions { WriteIndented = true });
                File.WriteAllText(settingsFile, json);
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error saving settings: {ex.Message}");
            }
        }

        private void MainForm_FormClosing(object sender, FormClosingEventArgs e)
        {
            SaveSettings();
            globalHotkey.Dispose();
        }

        protected override void WndProc(ref Message m)
        {
            globalHotkey.ProcessHotKeyMessage(ref m);
            base.WndProc(ref m);
        }
    }

    public class ClipboardItem
    {
        public string Id { get; set; } = Guid.NewGuid().ToString();
        public string Content { get; set; }
        public DateTime Timestamp { get; set; }
        public string Type { get; set; }
    }

    public class ClipboardData
    {
        public List<ClipboardItem> History { get; set; } = new List<ClipboardItem>();
        public List<ClipboardItem> Favorites { get; set; } = new List<ClipboardItem>();
        public Dictionary<string, Keys> Hotkeys { get; set; } = new Dictionary<string, Keys>();
    }

    public class HotkeyForm : Form
    {
        private TextBox hotkeyTextBox;
        public Keys SelectedHotkey { get; private set; }

        public HotkeyForm()
        {
            InitializeComponent();
        }

        private void InitializeComponent()
        {
            this.Text = "Set Hotkey";
            this.Size = new System.Drawing.Size(300, 150);
            this.StartPosition = FormStartPosition.CenterParent;
            this.FormBorderStyle = FormBorderStyle.FixedDialog;
            this.MaximizeBox = false;
            this.MinimizeBox = false;

            Label label = new Label();
            label.Text = "Press the hotkey combination:";
            label.Location = new System.Drawing.Point(20, 20);
            label.AutoSize = true;
            this.Controls.Add(label);

            hotkeyTextBox = new TextBox();
            hotkeyTextBox.Location = new System.Drawing.Point(20, 50);
            hotkeyTextBox.Size = new System.Drawing.Size(250, 25);
            hotkeyTextBox.ReadOnly = true;
            hotkeyTextBox.KeyDown += HotkeyTextBox_KeyDown;
            this.Controls.Add(hotkeyTextBox);

            Button okButton = new Button();
            okButton.Text = "OK";
            okButton.Location = new System.Drawing.Point(100, 80);
            okButton.DialogResult = DialogResult.OK;
            this.Controls.Add(okButton);

            Button cancelButton = new Button();
            cancelButton.Text = "Cancel";
            cancelButton.Location = new System.Drawing.Point(190, 80);
            cancelButton.DialogResult = DialogResult.Cancel;
            this.Controls.Add(cancelButton);

            this.AcceptButton = okButton;
            this.CancelButton = cancelButton;
        }

        private void HotkeyTextBox_KeyDown(object sender, KeyEventArgs e)
        {
            e.SuppressKeyPress = true;
            
            if (e.KeyCode != Keys.ShiftKey && e.KeyCode != Keys.ControlKey && e.KeyCode != Keys.Alt)
            {
                SelectedHotkey = e.KeyCode;
                
                if (e.Control)
                {
                    SelectedHotkey |= Keys.Control;
                }
                
                if (e.Shift)
                {
                    SelectedHotkey |= Keys.Shift;
                }
                
                if (e.Alt)
                {
                    SelectedHotkey |= Keys.Alt;
                }
                
                hotkeyTextBox.Text = SelectedHotkey.ToString();
            }
        }
    }

    public class GlobalHotkey : IDisposable
    {
        [DllImport("user32.dll")]
        private static extern bool RegisterHotKey(IntPtr hWnd, int id, uint fsModifiers, uint vk);

        [DllImport("user32.dll")]
        private static extern bool UnregisterHotKey(IntPtr hWnd, int id);

        private const int WM_HOTKEY = 0x0312;

        private IntPtr windowHandle;
        private Dictionary<string, int> hotkeyIds = new Dictionary<string, int>();
        private Dictionary<int, Action> hotkeyActions = new Dictionary<int, Action>();
        private int nextId = 1;

        public GlobalHotkey(IntPtr windowHandle)
        {
            this.windowHandle = windowHandle;
        }

        public void RegisterHotKey(string id, Keys hotkey, Action action)
        {
            uint modifiers = 0;
            
            if ((hotkey & Keys.Alt) == Keys.Alt)
                modifiers |= 0x0001; // MOD_ALT
                
            if ((hotkey & Keys.Control) == Keys.Control)
                modifiers |= 0x0002; // MOD_CONTROL
                
            if ((hotkey & Keys.Shift) == Keys.Shift)
                modifiers |= 0x0004; // MOD_SHIFT
                
            uint key = (uint)(hotkey & ~(Keys.Control | Keys.Shift | Keys.Alt));
            
            int hotkeyId = nextId++;
            bool registered = RegisterHotKey(windowHandle, hotkeyId, modifiers, key);
            
            if (registered)
            {
                hotkeyIds[id] = hotkeyId;
                hotkeyActions[hotkeyId] = action;
            }
        }

        public void UnregisterHotKey(string id)
        {
            if (hotkeyIds.TryGetValue(id, out int hotkeyId))
            {
                UnregisterHotKey(windowHandle, hotkeyId);
                hotkeyIds.Remove(id);
                hotkeyActions.Remove(hotkeyId);
            }
        }

        public void ProcessHotKeyMessage(ref Message m)
        {
            if (m.Msg == WM_HOTKEY)
            {
                int hotkeyId = m.WParam.ToInt32();
                if (hotkeyActions.TryGetValue(hotkeyId, out Action action))
                {
                    action?.Invoke();
                }
            }
        }

        public void Dispose()
        {
            foreach (int hotkeyId in hotkeyIds.Values)
            {
                UnregisterHotKey(windowHandle, hotkeyId);
            }
            
            hotkeyIds.Clear();
            hotkeyActions.Clear();
        }
    }

    static class Program
    {
        [STAThread]
        static void Main()
        {
            Application.EnableVisualStyles();
            Application.SetCompatibleTextRenderingDefault(false);
            Application.Run(new MainForm());
        }
    }
}