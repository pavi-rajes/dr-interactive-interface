To create an interactive web-interface to visualize neural data in the form of rastor plots, PSTH, location in mice brain etc. 

I want multiple tabs with each tab containing an interactive data exploration viewer 
- first tab: show brain regions that show context-dependent baseline firing shifts. Right side (3D interactive mice brain and dots representing the location of units with context dependent baseline shifts, Left side, select specific unit, brain region, analysis window, show rastor plots and PSTHs for auditory block and visual block)

- second tab: show brain regions where neurons fire specifically for rule-changes. Right side (3D interactive mice brain and dots representing the location of rule-updating units. Left side, select specific unit, brain region, context, analysis window, show rastor plots and PSTHs for auditory block and visual block)

Implementation notes:
- Use data from nwb_sessions to build the viewer but we should be able to swap or set the data location using a variable.
- Use Shiny Python package and app for interactive viewer. 
- any precomputed data must be saved in /data/precomputed. 
- all scripts must be inside /code and must be reproducible 

Agent implementation note:
- note down when Asta assistant is used for a task vs Claude is running the task 
- at the end, build me a flowchart that shows this workflow, which artifacts are produced at what steps, which decisions were made, etc. Label which harness/agent performed that step. 