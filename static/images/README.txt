This folder is reserved for static image assets (e.g. custom logos, favicons,
or screenshots) that you want to serve via Flask's static file handling.

The current UI does not require any image files to function — all icons are
rendered using Bootstrap Icons (loaded via CDN) and all illustrations/graphics
on the landing page are built with pure CSS/HTML (gradients, blobs, glass
panels), so the app works out of the box with this folder empty.

To add your own logo, place a file here (e.g. logo.png) and reference it in
templates like this:

    <img src="{{ url_for('static', filename='images/logo.png') }}" alt="Logo">
