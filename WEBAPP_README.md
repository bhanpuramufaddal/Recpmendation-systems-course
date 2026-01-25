# CS 329R Course Viewer - Web Application

A beautiful, interactive web application to view the CS 329R Recommendation Systems course materials systematically.

## Features

- **Organized Navigation**: Browse through all 19 weeks of course content with an intuitive sidebar
- **Markdown Rendering**: Full support for GitHub-flavored markdown
- **LaTeX Support**: Mathematical formulas rendered beautifully using KaTeX
- **Code Highlighting**: Syntax highlighting for Python code examples
- **Responsive Design**: Clean, readable interface optimized for learning
- **Progressive Navigation**: Expand weeks to see individual topics

## Quick Start

### Option 1: Using Python (Recommended)

Simply run the provided Python server script:

```bash
python3 start_server.py
```

This will:
1. Start a local web server on port 8000
2. Automatically open the course viewer in your default browser
3. Display the URL: `http://localhost:8000/index.html`

Press `Ctrl+C` to stop the server when done.

### Option 2: Direct File Access

You can also open `index.html` directly in your browser, though you may encounter CORS issues with some browsers. Use the Python server method for the best experience.

## Usage

1. **Navigate by Week**: Click on any week in the sidebar to expand it
2. **View Topics**: Click on individual topics to load the content
3. **Course Overview**: The default view shows the main README with course structure
4. **Read Content**: All markdown, LaTeX formulas, tables, and code are rendered beautifully

## Course Structure

The viewer automatically organizes content from:
- Main course README
- 19 weekly modules (week-01 through week-19)
- Individual topic files within each week
- Practice problems, code examples, and paper summaries

## Technical Details

### Technologies Used

- **Marked.js**: Markdown parsing and rendering
- **KaTeX**: Fast LaTeX rendering
- **Highlight.js**: Code syntax highlighting
- **Vanilla JavaScript**: No framework dependencies
- **Responsive CSS**: Mobile-friendly design

### Supported Content

- Markdown headings, lists, tables, blockquotes
- LaTeX math (inline with `$...$` and display with `$$...$$`)
- Code blocks with syntax highlighting
- Links, images, and embedded content
- Horizontal rules and formatted text

## Browser Compatibility

Works best with modern browsers:
- Chrome/Chromium
- Firefox
- Safari
- Edge

## Customization

You can customize the appearance by editing the `<style>` section in `index.html`:
- Color scheme (currently dark sidebar + light content)
- Font sizes and families
- Spacing and layout
- Sidebar width

## Troubleshooting

**Problem**: Content not loading
- **Solution**: Use the Python server script instead of opening the HTML file directly

**Problem**: LaTeX not rendering
- **Solution**: Check your internet connection (KaTeX is loaded from CDN)

**Problem**: Code not highlighted
- **Solution**: Check your internet connection (Highlight.js is loaded from CDN)

**Problem**: Port 8000 already in use
- **Solution**: Edit `start_server.py` and change the `PORT` variable to another number (e.g., 8080)

## Features in Detail

### Navigation
- Expandable/collapsible week sections
- Active state highlighting
- Breadcrumb navigation
- Smooth scrolling

### Content Display
- Maximum width for optimal readability
- Professional typography
- Clear visual hierarchy
- Syntax-highlighted code blocks

### Math Rendering
All mathematical formulas are rendered using KaTeX, including:
- Inline equations: `$x^2 + y^2 = z^2$`
- Display equations: `$$\frac{-b \pm \sqrt{b^2-4ac}}{2a}$$`
- Complex formulas with matrices, summations, etc.

## Future Enhancements

Potential improvements:
- Search functionality
- Bookmarking/favorites
- Dark mode toggle
- Print-friendly CSS
- Offline support with service worker
- Export to PDF

## License

This viewer is part of the CS 329R course materials.

---

**Enjoy learning about Recommendation Systems!**
