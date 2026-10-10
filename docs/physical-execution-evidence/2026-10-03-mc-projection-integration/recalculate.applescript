on run argv
    set targetPath to item 1 of argv
    tell application "Microsoft Excel"
        if (count of workbooks) is not 0 then error "Excel has another workbook open; full rebuild cancelled to preserve it."
        open workbook workbook file name targetPath
        set targetBook to workbook "conformance.xlsx"
        calculate full rebuild
        calculate full
        save targetBook
        set excelVersion to version
        close targetBook saving yes
        return excelVersion
    end tell
end run
