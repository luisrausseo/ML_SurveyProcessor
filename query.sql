SELECT LTRIM(SUBSTRING([text], CHARINDEX('>', [text])+5, 1000)) as text
       ,CONCAT('__label__', label) as label 
FROM (
	  SELECT LTRIM(RTRIM(SUBSTRING([mrALLDESCRIPTIONS], 0, CHARINDEX('ORIGINAL ISSUE =', [mrALLDESCRIPTIONS])))) as text 
	  ,[Customer__bComments__bRating] as label
	  FROM [Footprints].[dbo].[MASTER14] 
	  WHERE NOT [Customer__bComments__bRating] IN (0, '__u1')
	 ) a