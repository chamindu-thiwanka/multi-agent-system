# ================================================================
# frontend/Dockerfile
# ================================================================
# Serves the single HTML file using a lightweight HTTP server.
#
# WHY NOT JUST OPEN index.html DIRECTLY?
# When you open a local file (file:///...), browsers block
# fetch() calls to localhost due to CORS security restrictions.
# Running it through a proper HTTP server avoids this.
#
# WHY nginx?
# nginx is a battle-tested, extremely lightweight web server.
# The nginx:alpine image is only ~20MB.
# It serves static files (HTML, CSS, JS) with zero configuration.
# ================================================================

# Use the official nginx alpine image
# alpine = tiny Linux distribution (~5MB base)
# nginx:alpine = nginx web server on alpine (~20MB total)
FROM nginx:alpine

# Remove the default nginx welcome page
# We're replacing it with our own index.html
RUN rm /usr/share/nginx/html/*

# Copy our HTML file into nginx's serving directory
# nginx automatically serves files from /usr/share/nginx/html/
COPY index.html /usr/share/nginx/html/index.html

# Expose port 80 (nginx's default HTTP port)
# docker-compose maps this to port 3000 on your laptop
EXPOSE 80

# nginx starts automatically with the container
# No CMD needed - the base image already handles this