WILD HEARTS Shader Pre-compiler  v1.12.0
=========================================

WHAT IT DOES
WILD HEARTS stutters the first time you see many effects, because the game
prepares ("compiles") the shader for each one at that moment. This mod
prepares all of the game's shaders in the background when the game starts,
so they are ready before you need them.

- Works with NVIDIA, AMD and Intel graphics cards on Windows 10 and 11.
- It doesn't change how the game looks. All you see is a small progress
  line in the top-left corner, which goes away when it's done.
- Nothing is downloaded. Everything it needs is already in your game files.

INSTALL
1. In Steam, right-click WILD HEARTS > Manage > Browse local files.
2. Open the folder  00_game\target_origin\ex  (the one with WILD HEARTS.exe).
3. Copy dxgi.dll and the WH_SHADERS folder from this zip into it, right
   next to WILD HEARTS.exe (not into a subfolder).
4. Start the game. A line of text in the top-left corner means it's working.

WHAT YOU'LL SEE
The first start does a lot of work once. You can play while it runs. The
progress line says what it's doing. The main steps:

  deep scan of every game archive
      Looks through the game files for every shader. About a minute on an
      SSD, longer on a hard drive. Only happens again after a game or mod
      update.
  learning render settings from the game
      Watches how the game draws. Loading into the game speeds this up.
      A few minutes at most.
  reading shader containers
      A quick pass over the shaders it found.
  rebuilding recorded pipelines
      Rebuilds what is already known. Usually a few seconds.
  compiling shaders
      The main job. The first time, this can take 10 minutes or more,
      depending on your CPU.
  done
      Shows for a few seconds, then the line goes away.

Later starts are quick: it rebuilds what it prepared before and only
compiles what's new.

SETTINGS
Open WH_SHADERS\WH_PSO.ini with Notepad, change a value, save, and restart
the game. The ones you might want:
- Hide the progress line: under [Overlay], set Enabled=0.
- Use a fixed number of CPU threads: under [Performance], set ReplayThreads
  and PrecompileThreads to a number, e.g. 4. By default the mod uses the
  CPU threads the game isn't using.
- Write a log for troubleshooting: under [Logging], set Enabled=1.

UPDATE
Copy the new files over the old ones and choose Replace. Your settings go
back to the defaults.

UNINSTALL
Delete dxgi.dll and the WH_SHADERS folder. If the folder
%LOCALAPPDATA%\WH_SHADERS exists (paste that into the Explorer address bar
to check), delete it too.

QUESTIONS
Will it still stutter sometimes?
  Much less. A few effects the mod couldn't prepare may still stutter once.
  The mod remembers those and prepares them at every start after that.

Why is my CPU busier than usual at first?
  That's the compiling. By default the mod uses CPU threads the game isn't
  using and leaves the game at least a quarter of them, so the game stays
  playable.

Why does it compile everything again after a driver update?
  A new driver throws away the shaders it had saved. The mod notices and
  prepares them again at the next start. The same happens after changing
  your graphics card or clearing the shader cache.

How much disk space does it need?
  The graphics driver's shader cache for WILD HEARTS grows to about 6 to 7
  GB. Make sure the driver may keep that much, or it throws shaders away
  and they stutter again:
  - NVIDIA: NVIDIA Control Panel > Manage 3D settings > Shader Cache Size
    > Unlimited.
  - AMD: keep Shader Cache on in AMD Software (Gaming > Graphics).

Can I use it with other mods?
  Yes, except other mods that are also a dxgi.dll (for example ReShade
  installed as dxgi.dll, Special K, OptiScaler). Only one dxgi.dll can be
  in the folder: whichever you copy last replaces the other.

Where does it keep its files?
  In the WH_SHADERS folder next to the game. In the rare case that Windows
  won't let the game write there (a protected folder outside Steam), the
  progress and the log go to a folder inside %LOCALAPPDATA%\WH_SHADERS
  instead (paste that into the Explorer address bar). The settings file
  stays next to the game; to change it there, run Notepad as administrator.

Steam Deck or Linux?
  Untested. To try anyway, set the game's launch options in Steam to:
  WINEDLLOVERRIDES="dxgi=n,b" %command%

PROBLEMS?
- No progress line: check that dxgi.dll is right next to WILD HEARTS.exe,
  that no other mod replaced it, and that Enabled=1 under [Overlay].
- The game crashes or won't start: delete dxgi.dll to rule the mod out. If
  the game crashes twice while the same shader is being prepared, the mod
  skips that shader until your next driver update.
- To report a problem: set Enabled=1 under [Logging] in WH_PSO.ini, play
  once, and send the file wh_pso.log from WH_SHADERS\logs (or from the
  folder inside %LOCALAPPDATA%\WH_SHADERS) with a short description.
