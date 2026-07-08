"""Small shared Tkinter UI helpers."""
import tkinter as tk
from tkinter import ttk


def scrolled(parent, factory, vertical=True, horizontal=True, pack=None):
    """Build a scroll-wrapped widget.

    Creates a container frame packed into `parent`, builds the widget inside it
    via `factory(container)`, lays out the widget with grid, and attaches
    vertical/horizontal scrollbars that appear whenever the content exceeds the
    visible area. Returns the widget (not the container).

    factory: callable(container) -> a widget supporting yview/xview
             (Treeview, Text, Listbox, Canvas).
    pack:    optional dict of pack options for the container
             (default fills and expands).
    """
    container = ttk.Frame(parent)
    container.pack(**(pack if pack is not None else {"fill": "both", "expand": True,
                                                     "padx": 4, "pady": 4}))
    container.rowconfigure(0, weight=1)
    container.columnconfigure(0, weight=1)

    widget = factory(container)
    widget.grid(row=0, column=0, sticky="nsew")

    if vertical:
        vs = ttk.Scrollbar(container, orient="vertical", command=widget.yview)
        widget.configure(yscrollcommand=vs.set)
        vs.grid(row=0, column=1, sticky="ns")
    if horizontal:
        hs = ttk.Scrollbar(container, orient="horizontal", command=widget.xview)
        widget.configure(xscrollcommand=hs.set)
        hs.grid(row=1, column=0, sticky="ew")

    return widget
