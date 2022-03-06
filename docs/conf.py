# -*- coding: utf-8 -*-
#
# Kaggler documentation build configuration file, created by
# sphinx-quickstart on Tue Feb 10 04:55:59 2015.
#
# This file is execfile()d with the current directory set to its
# containing dir.
#
# Note that not all possible configuration values are present in this
# autogenerated file.
#
# All configuration values have a default; values that are commented out
# serve to show the default.

import sys
import os
import matplotlib
matplotlib.use('agg')

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
sys.path.insert(0, os.path.abspath("../.."))

import kaggler            # noqa
import sphinx_rtd_theme   # noqa

# -- General configuration ------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinxcontrib.napoleon',
    'sphinx.ext.autodoc',
    'sphinx.ext.doctest',
    'sphinx.ext.intersphinx',
    'sphinx.ext.todo',
    'sphinx.ext.coverage',
    'sphinx.ext.mathjax',
    'sphinx.ext.viewcode',
]

autodoc_mock_imports = ["_tkinter"]


# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The suffix of source filenames.
source_suffix = '.rst'

# The encoding of source files.

# The master toctree document.
master_doc = 'index'

# General information about the project.
project = u'Kaggler'
copyright = u'2019, Jeong-Yoon Lee'

# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.
#
# The short X.Y version.
version = kaggler.__version__
# The full version, including alpha/beta/rc tags.
release = kaggler.__version__

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.

# There are two options for replacing |today|: either, you set today to some
# non-false value, then it is used:
# Else, today_fmt is used as the format for a strftime call.

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = ['_build']

# The reST default role (used for this markup: `text`) to use for all
# documents.

# If true, '()' will be appended to :func: etc. cross-reference text.

# If true, the current module name will be prepended to all description
# unit titles (such as .. function::).

# If true, sectionauthor and moduleauthor directives will be shown in the
# output. They are ignored by default.

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'

# A list of ignored prefixes for module index sorting.

# If true, keep warnings as "system message" paragraphs in the built documents.


# -- Options for HTML output ----------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme = 'sphinx_rtd_theme'

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.

# Add any paths that contain custom themes here, relative to this directory.
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]

# The name for this set of Sphinx documents.  If None, it defaults to
# "<project> v<release> documentation".

# A shorter title for the navigation bar.  Default is the same as html_title.

# The name of an image file (relative to this directory) to place at the top
# of the sidebar.

# The name of an image file (within the static path) to use as favicon of the
# docs.  This file should be a Windows icon file (.ico) being 16x16 or 32x32
# pixels large.

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# Add any extra paths that contain custom files (such as robots.txt or
# .htaccess) here, relative to this directory. These files are copied
# directly to the root of the documentation.

# If not '', a 'Last updated on:' timestamp is inserted at every page bottom,
# using the given strftime format.

# If true, SmartyPants will be used to convert quotes and dashes to
# typographically correct entities.

# Custom sidebar templates, maps document names to template names.

# Additional templates that should be rendered to pages, maps page names to
# template names.

# If false, no module index is generated.

# If false, no index is generated.

# If true, the index is split into individual pages for each letter.

# If true, links to the reST sources are added to the pages.

# If true, "Created using Sphinx" is shown in the HTML footer. Default is True.

# If true, "(C) Copyright ..." is shown in the HTML footer. Default is True.

# If true, an OpenSearch description file will be output, and all pages will
# contain a <link> tag referring to it.  The value of this option must be the
# base URL from which the finished HTML is served.

# This is the file name suffix for HTML files (e.g. ".xhtml").

# Output file base name for HTML help builder.
htmlhelp_basename = 'Kagglerdoc'


# -- Options for LaTeX output ---------------------------------------------

latex_elements = {
  # The paper size ('letterpaper' or 'a4paper').

  # The font size ('10pt', '11pt' or '12pt').

  # Additional stuff for the LaTeX preamble.
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = [
  ('index', 'Kaggler.tex', u'Kaggler Documentation',
   u'Jeong-Yoon Lee', 'manual'),
]

# The name of an image file (relative to this directory) to place at the top of
# the title page.

# For "manual" documents, if this is true, then toplevel headings are parts,
# not chapters.

# If true, show page references after internal links.

# If true, show URL addresses after external links.

# Documents to append as an appendix to all manuals.

# If false, no module index is generated.


# -- Options for manual page output ---------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [
    ('index', 'kaggler', u'Kaggler Documentation',
     [u'Jeong-Yoon Lee'], 1)
]

# If true, show URL addresses after external links.

# -- Options for Texinfo output -------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
  ('index', 'Kaggler', u'Kaggler Documentation',
   u'Jeong-Yoon Lee', 'Kaggler', 'One line description of project.',
   'Miscellaneous'),
]

# Documents to append as an appendix to all manuals.

# If false, no module index is generated.

# How to display URL addresses: 'footnote', 'no', or 'inline'.

# If true, do not generate a @detailmenu in the "Top" node's menu.


# Example configuration for intersphinx: refer to the Python standard library.
intersphinx_mapping = {'http://docs.python.org/': None}

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
