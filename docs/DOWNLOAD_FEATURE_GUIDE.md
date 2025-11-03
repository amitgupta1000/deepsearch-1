# Download Feature Guide

## Overview

Users can now choose between downloading research reports as **Text (.txt)** or **PDF (.pdf)** files.

## Changes Made

### Backend Changes

#### 1. Updated Download API Endpoint
- **Endpoint**: `GET /api/research/{session_id}/download`
- **New Parameter**: `format` (query parameter)
- **Accepted Values**: `txt` or `pdf`
- **Default**: `txt` if not specified

**Examples:**
```
GET /api/research/abc123/download?format=txt
GET /api/research/abc123/download?format=pdf
```

#### 2. File Path Logic
- **TXT files**: Uses original report filename from session
- **PDF files**: Replaces `.txt` extension with `.pdf` for the same base filename
- **Error handling**: Returns appropriate HTTP errors for missing files or invalid formats

### Frontend Changes

#### 1. UI Updates
- **Before**: Single "Download" button
- **After**: Two separate buttons:
  - "Download TXT" with document icon
  - "Download PDF" with download icon

#### 2. Download Logic
- **With Session ID**: Uses server-side download via API
- **Without Session ID**: Falls back to client-side text download
- **Error Handling**: Shows user-friendly error messages and fallbacks

#### 3. Type System Updates
- Added `sessionId?: string` to `ResearchResult` interface
- Updated ResearchContext to include sessionId in results

## How It Works

### For Session-Based Downloads (Web App)
1. User completes a research request
2. Backend generates both `.txt` and `.pdf` files
3. Frontend receives sessionId with the result
4. User clicks "Download TXT" or "Download PDF"
5. Frontend calls API with appropriate format parameter
6. Backend serves the requested file format

### For Direct Downloads (Export Version)
1. User completes a research request  
2. Frontend receives result content without sessionId
3. User clicks download buttons
4. System falls back to client-side text download
5. PDF option shows error message (since no server-side PDF generation)

## File Structure

```
IntelliSearchReport_YYYYMMDD_HHMMSS.txt  # Text report
IntelliSearchReport_YYYYMMDD_HHMMSS.pdf  # PDF report (auto-generated)
```

## Error Handling

### Invalid Format
- **Request**: `?format=doc`
- **Response**: `400 Bad Request - Format must be either 'txt' or 'pdf'`

### Missing File
- **PDF not found**: `404 Not Found - Report PDF file not found`
- **TXT not found**: `404 Not Found - Report text file not found`

### Session Issues
- **Invalid session**: `404 Not Found - Research session not found`
- **Incomplete research**: `400 Bad Request - Report not available for download`

## User Experience

1. **Clear Choice**: Users see two distinct download options
2. **Immediate Feedback**: Download starts immediately on button click
3. **Fallback Support**: Graceful degradation when server downloads aren't available
4. **Error Messages**: Clear feedback when downloads fail

## Technical Implementation

### Backend API
```python
@app.get("/api/research/{session_id}/download")
async def download_report(session_id: str, format: str = "txt"):
    # Validate format
    if format not in ["txt", "pdf"]:
        raise HTTPException(status_code=400, detail="Format must be either 'txt' or 'pdf'")
    
    # Determine file path based on format
    if format == "pdf":
        file_path = text_path.replace(".txt", ".pdf")
        media_type = "application/pdf"
    else:
        file_path = original_text_path
        media_type = "text/plain"
    
    return FileResponse(path=file_path, media_type=media_type)
```

### Frontend Implementation
```typescript
const downloadReportAs = async (format: 'txt' | 'pdf') => {
  if (!result.sessionId) {
    // Fallback to local download
    downloadReport();
    return;
  }

  const response = await fetch(`/api/research/${result.sessionId}/download?format=${format}`);
  // Handle response and trigger download
};
```

## Benefits

1. **User Choice**: Users can pick their preferred format
2. **Flexibility**: Works with or without session management
3. **Backward Compatible**: Existing functionality preserved
4. **Professional Output**: PDF option provides better formatted reports
5. **Error Resilient**: Multiple fallback mechanisms ensure users can always get content